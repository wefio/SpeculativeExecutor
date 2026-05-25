import json
import asyncio
from tools import ToolRegistry
from provider import DeepSeekProvider
from speculative import SpeculativeExecutor, SpecConfig, build_signatures
from strategies import get_system_prompt, get_max_iterations


class Agent:
    def __init__(self, provider: DeepSeekProvider, tools: ToolRegistry,
                 strategy: str = "simple", speculative: bool = False,
                 spec_level: str = "auto", spec_config: SpecConfig | None = None):
        self.provider = provider
        self.tools = tools
        self.strategy = strategy
        self.max_iterations = get_max_iterations(strategy)
        if speculative:
            config = spec_config or SpecConfig(level=spec_level)
            self.speculator = SpeculativeExecutor(provider, tools, config=config)
        else:
            self.speculator = None
        self.messages: list[dict] = [
            {"role": "system", "content": get_system_prompt(strategy, build_signatures(tools))}
        ]

    async def run(self, user_input: str) -> str:
        if self.speculator:
            self.speculator.clear()
        if self.strategy == "react":
            return await self._run_react(user_input)
        elif self.strategy == "plan_solve":
            return await self._run_plan_solve(user_input)
        else:
            return await self._run_simple(user_input)

    # ── Simple ──────────────────────────────────────────────────────────

    async def _run_simple(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        for i in range(self.max_iterations):
            print(f"\n{'─' * 40}")
            print(f"[Iter {i+1}] Agent thinking")

            response = await self._call_model()

            if response.reasoning_content:
                print(f"[Thinking]\n{response.reasoning_content[:500]}")

            if response.content:
                print(f"[Content] {response.content[:300]}")

            if not response.tool_calls:
                self._append_assistant(response)
                return response.content or ""

            self._append_assistant(response)
            await self._execute_tools_parallel(response)

        return "(max iterations reached)"

    # ── ReAct ───────────────────────────────────────────────────────────

    async def _run_react(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        for step in range(self.max_iterations):
            print(f"\n{'─' * 40}")
            print(f"[ReAct Step {step + 1}]")
            response = await self._call_model()

            if response.reasoning_content:
                print(f"[Thinking]\n{response.reasoning_content[:500]}")

            if response.content:
                print(f"[Thought] {response.content[:300]}")

            if not response.tool_calls:
                self._append_assistant(response)
                return response.content or ""

            self._append_assistant(response)
            observations = await self._execute_tools_parallel(response)

            obs_text = "\n".join(f"[Observation] {o[:300]}" for o in observations)
            self.messages.append({
                "role": "user",
                "content": f"Observation:\n{obs_text}\n\nContinue with your next Thought, or give Final Answer."
            })

        return "(max iterations reached)"

    # ── Plan-and-Solve ──────────────────────────────────────────────────

    async def _run_plan_solve(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        print(f"\n{'─' * 40}")
        print("[Planning]")
        plan_response = await self._call_model()

        if plan_response.reasoning_content:
            print(f"[Thinking]\n{plan_response.reasoning_content[:500]}")

        if plan_response.content:
            print(f"[Plan]\n{plan_response.content[:500]}")

        if plan_response.tool_calls:
            self._append_assistant(plan_response)
            await self._execute_tools_parallel(plan_response)
        else:
            self._append_assistant(plan_response)
            self.messages.append({
                "role": "user",
                "content": "Good plan. Now execute it step by step. Call tools as needed."
            })

        for step in range(self.max_iterations - 1):
            print(f"\n{'─' * 40}")
            print(f"[Solving Step {step + 1}]")
            response = await self._call_model()

            if response.reasoning_content:
                print(f"[Thinking]\n{response.reasoning_content[:500]}")

            if response.content:
                print(f"[Content] {response.content[:300]}")

            if not response.tool_calls:
                self._append_assistant(response)
                return response.content or ""

            self._append_assistant(response)
            await self._execute_tools_parallel(response)

        return "(max iterations reached)"

    # ── Shared helpers ──────────────────────────────────────────────────

    async def _call_model(self):
        """Call main model. Speculative prediction runs as background task."""
        if self.speculator:
            self.speculator.predict(self.messages, self._last_user_query())

        return await self.provider.chat(
            messages=self.messages,
            tools=self.tools.get_definitions(),
        )

    def _append_assistant(self, response):
        msg = {"role": "assistant", "content": response.content}
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response.tool_calls
            ]
        if response.reasoning_content:
            msg["reasoning_content"] = response.reasoning_content
        self.messages.append(msg)

    async def _execute_tools_parallel(self, response) -> list[str]:
        """Execute tool calls in parallel. Check speculative cache for each."""
        tool_calls = list(response.tool_calls)

        # Check cache for each tool call
        cached_results = {}
        for tc in tool_calls:
            name = tc.function.name
            args = tc.function.arguments
            print(f"\n[Tool Call] {name}({args})")
            if self.speculator and self.speculator.match(name, args):
                result = self.speculator.cached_result(name, args)
                if result:
                    cached_results[tc.id] = result
                    print(f"[Spec] HIT (cache matched)")
                    print(f"[Tool Result] {result[:500]}")

        # Execute uncached tools in parallel
        uncached = [tc for tc in tool_calls if tc.id not in cached_results]

        async def exec_one(tc):
            result = self.tools.execute(tc.function.name, tc.function.arguments)
            print(f"[Tool Result] {result[:500]}")
            return tc.id, result

        if uncached:
            exec_results = await asyncio.gather(*[exec_one(tc) for tc in uncached])
        else:
            exec_results = []

        # Merge results in original order
        all_results = []
        for tc in tool_calls:
            if tc.id in cached_results:
                result = cached_results[tc.id]
            else:
                result = next(r for tid, r in exec_results if tid == tc.id)
            all_results.append(result)

            self.messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        return all_results

    def _last_user_query(self) -> str:
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                return msg["content"]
        return ""
