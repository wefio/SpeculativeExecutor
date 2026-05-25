"""Speculative executor — predicts tool calls via prefix continuation."""
import json
import re
import time
import asyncio
from dataclasses import dataclass

from .types import PredictionResult, PredictionStats
from . import matcher


@dataclass
class SpecConfig:
    """Configuration for speculative execution."""
    level: str = "auto"              # "auto" | "prefix_fim" | "prefix" | "disabled"
    max_failures: int = 3            # consecutive failures before disabling
    backoff_seconds: int = 300       # rate limit backoff
    context: str = ""                # environment context
    record_text: bool = False        # store prefix raw text
    system_prompt: str = ""          # custom system prompt


_DEFAULT_SYSTEM = (
    'You are an assistant with tools. Use tools when needed.\n'
    'If no tool is needed, respond with: {"tool": null}\n'
    'Available tools (call as JSON: {"tool": "name", "arguments": {...}}):'
)


def build_signatures(tools) -> str:
    """Build compact Python-style tool signatures from a ToolRegistry."""
    lines = []
    for t in tools.get_definitions():
        fn = t["function"]
        desc = fn.get("description", "")
        params = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        parts = []
        for k, v in params.items():
            ptype = {"string": "str", "integer": "int", "boolean": "bool",
                     "number": "float"}.get(v.get("type", ""), "str")
            if k in required:
                parts.append(f"{k}: {ptype}")
            else:
                default = '""' if ptype == "str" else "None"
                parts.append(f"{k}: {ptype} = {default}")
        sig = f"{fn['name']}({', '.join(parts)})"
        lines.append(f"{sig} — {desc}" if desc else sig)
    return "\n".join(lines)


class SpeculativeExecutor:
    """Predicts tool calls in parallel with the main model.

    Supports multiple providers (models) racing simultaneously.
    The first correct prediction wins.

    Usage:
        # Single provider
        spec = SpeculativeExecutor(provider, tools, config)

        # Multiple providers (race)
        spec = SpeculativeExecutor([provider1, provider2, provider3], tools, config)
    """

    def __init__(self, provider, tools, config: SpecConfig | None = None):
        self.providers = provider if isinstance(provider, list) else [provider]
        self.tools = tools
        self.config = config or SpecConfig()
        self.sigs = build_signatures(tools)
        self.level = self._detect_level()

        self._caches: list[PredictionResult] = []
        self._failures = 0
        self._backoff_until = 0.0
        self.stats: list[PredictionStats] = []
        self.context = self.config.context  # shortcut for external setting

    # ── Public API ────────────────────────────────────────────────────

    def predict(self, messages: list[dict], user_query: str = ""):
        """Fire speculative prediction as background task. Non-blocking."""
        if self.level == "disabled":
            return
        if self._failures >= self.config.max_failures > 0:
            return
        if time.time() < self._backoff_until:
            return
        asyncio.create_task(self._run(messages, user_query))

    def match(self, actual_name: str, actual_args: str) -> bool:
        """Check if any cached prediction matches the actual tool call."""
        for cached in self._caches:
            if cached.matches(actual_name, actual_args):
                return True
        return False

    def match_gt(self, actual_name: str, actual_args: str,
                 ground_truth: list[dict]) -> bool:
        """Check if cached prediction matches BFCL ground truth."""
        for cached in self._caches:
            if matcher.match_ground_truth(cached.tool_name, cached.arguments, ground_truth):
                return True
        return False

    def cached_result(self, name: str = "", args: str = "") -> str | None:
        """Get cached tool result. If name/args given, find exact match; else return first."""
        if not self._caches:
            return None
        if name:
            for cached in self._caches:
                if cached.matches(name, args):
                    return cached.result
            return None
        return self._caches[0].result

    def clear(self):
        """Clear caches and stats (call between turns)."""
        self._caches.clear()
        self.stats.clear()

    @property
    def predictions(self) -> list[dict]:
        """Get all cached predictions as dicts (for external inspection)."""
        return [{"name": c.tool_name, "arguments": c.arguments,
                 "result": c.result, "level": c.level} for c in self._caches]

    # ── Internal ──────────────────────────────────────────────────────

    def _detect_level(self) -> str:
        if self.config.level != "auto":
            return self.config.level
        for p in self.providers:
            if hasattr(p, "fim_complete"):
                return "prefix_fim"
        for p in self.providers:
            if hasattr(p, "prefix_complete"):
                return "prefix"
        return "disabled"

    async def _run(self, messages: list[dict], user_query: str):
        """Run prediction on all providers, use first result."""
        stats = PredictionStats()
        t0 = time.perf_counter()

        if len(self.providers) == 1:
            # Single provider: direct call
            try:
                await self._predict(self.providers[0], messages, user_query, stats)
                self._failures = 0
            except Exception as e:
                self._failures += 1
                stats.error = str(e)
                stats.total_duration = time.perf_counter() - t0
                self.stats.append(stats)
                if "429" in str(e) or "rate" in str(e).lower():
                    self._backoff_until = time.time() + self.config.backoff_seconds
                return
        else:
            # Multiple providers: race (FIRST_COMPLETED)
            async def try_provider(p):
                s = PredictionStats()
                await self._predict(p, messages, user_query, s)
                return s

            tasks = [asyncio.create_task(try_provider(p)) for p in self.providers]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()

            winner = done.pop().result()
            if winner.error:
                self._failures += 1
            else:
                self._failures = 0
            stats = winner

        stats.total_duration = time.perf_counter() - t0
        self.stats.append(stats)

    async def _predict(self, provider, messages: list[dict], user_query: str,
                       stats: PredictionStats):
        # Build context: explicit config.context or extract from messages
        context = self.context or self.config.context or self._extract_history(messages)

        system = self.config.system_prompt or _DEFAULT_SYSTEM
        prefix_msgs = [
            {"role": "system", "content": f"{system}\n{self.sigs}"},
        ]
        if context:
            prefix_msgs.append({"role": "assistant", "content": context})
        prefix_msgs.append({"role": "user", "content": user_query})

        # Stage 1: prefix continuation
        t1 = time.perf_counter()
        stage1 = await provider.prefix_complete(
            prefix_msgs, '{"tool":', return_usage=False)
        stats.prefix_duration = time.perf_counter() - t1

        full_text = '{"tool":' + stage1
        if self.config.record_text:
            stats.prefix_text = full_text

        tool_name, args_json = self._parse(full_text)
        if not tool_name:
            return

        # Stage 2: FIM for empty args
        if args_json == "{}" and self.level == "prefix_fim":
            t2 = time.perf_counter()
            args_json = await self._fim_fill(provider, tool_name, full_text)
            stats.fim_duration = time.perf_counter() - t2
            if self.config.record_text:
                stats.fim_text = args_json

        # Execute speculatively and cache
        result = self.tools.execute(tool_name, args_json)
        level_tag = "prefix_fim" if (args_json != "{}" and self.level == "prefix_fim") else "prefix"

        stats.tool_name = tool_name
        stats.args_json = args_json
        stats.level = level_tag

        self._caches.append(PredictionResult(tool_name, args_json, result, level_tag))

    def _parse(self, text: str) -> tuple[str | None, str | None]:
        """Parse tool call from prefix output."""
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            raw = json_match.group()
            if raw.count('{') == raw.count('}'):
                parsed = json.loads(raw)
                if "tool" in parsed and parsed["tool"] is None:
                    return None, None
                if "tool" in parsed and isinstance(parsed["tool"], str):
                    return parsed["tool"], json.dumps(parsed.get("arguments", {}))
                if "name" in parsed:
                    return parsed["name"], json.dumps(parsed.get("arguments", {}))

        if re.search(r'"tool"\s*:\s*null', text):
            return None, None

        # Fallback: known tool names
        for name in ["python_execute", "shell_execute", "file_read",
                      "file_write", "http_request"]:
            if name in text:
                for key in ["arguments", "args", "params"]:
                    m = re.search(rf'"{key}"\s*:\s*(\{{[\s\S]*?\}})', text)
                    if m and m.group(1).count('{') == m.group(1).count('}'):
                        return name, m.group(1)
                return name, "{}"

        return None, None

    async def _fim_fill(self, provider, tool_name: str, prefix_output: str) -> str:
        """FIM Stage 2: fill empty arguments."""
        args_start = prefix_output.find('"arguments"')
        if args_start == -1:
            return "{}"

        fim_prefix = prefix_output[:args_start] + '"arguments": {'
        fim_suffix = "}}"

        fim_result = await provider.fim_complete(
            prefix=fim_prefix, suffix=fim_suffix, max_tokens=128)

        full = fim_prefix + fim_result + fim_suffix
        if full.count('{') == full.count('}'):
            parsed = json.loads(full)
            return json.dumps(parsed.get("arguments", {}))

        fixed = re.sub(r',\s*}}', '}}', full)
        if fixed.count('{') == fixed.count('}'):
            parsed = json.loads(fixed)
            return json.dumps(parsed.get("arguments", {}))

        return "{}"

    @staticmethod
    def _extract_history(messages: list[dict]) -> str:
        """Extract tool call history from messages (mechanical, zero cost)."""
        lines = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    lines.append(f"Called {fn.get('name', '')}({fn.get('arguments', '')})")
        return "\n".join(lines)
