import re
import asyncio
from openai import AsyncOpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')


def _sanitize(obj):
    """Remove surrogate characters that break UTF-8 encoding."""
    if isinstance(obj, str):
        return _SURROGATE_RE.sub('', obj)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


class DeepSeekProvider:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        self.model = DEEPSEEK_MODEL
        self.chat_usage = None
        self.prefix_usage = None

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   tool_choice: str | dict | None = None):
        kwargs = {"model": self.model, "messages": _sanitize(messages)}
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        response = await self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        msg._usage = response.usage
        self.chat_usage = response.usage
        return msg

    async def chat_stream(self, messages: list[dict], tools: list[dict] | None = None):
        kwargs = {"model": self.model, "messages": _sanitize(messages), "stream": True}
        if tools:
            kwargs["tools"] = tools

        stream = await self.client.chat.completions.create(**kwargs)

        full_text = ""
        reasoning_content = ""
        tool_calls = []
        current_tool = None

        async for chunk in stream:
            delta = chunk.choices[0].delta

            rc = getattr(delta, "reasoning_content", None)
            if rc and rc != "None":
                reasoning_content += rc
                continue

            if delta.content:
                print(delta.content, end="", flush=True)
                full_text += delta.content

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.index is not None:
                        while len(tool_calls) <= tc.index:
                            tool_calls.append({"id": "", "function": {"name": "", "arguments": ""}})
                        current_tool = tool_calls[tc.index]
                    if tc.id:
                        current_tool["id"] = tc.id
                    if tc.function and tc.function.name:
                        current_tool["function"]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        current_tool["function"]["arguments"] += tc.function.arguments

        print()

        return type("Message", (), {
            "content": full_text or None,
            "reasoning_content": reasoning_content or None,
            "tool_calls": [
                type("ToolCall", (), {
                    "id": t["id"],
                    "type": "function",
                    "function": type("Func", (), {
                        "name": t["function"]["name"],
                        "arguments": t["function"]["arguments"],
                    }),
                })
                for t in tool_calls
            ] if tool_calls else None,
        })()

    async def prefix_complete(self, messages: list[dict], assistant_prefix: str, return_usage: bool = False):
        """Prefix continuation — pass assistant prefix, get continuation."""
        clean_msgs = []
        for msg in messages:
            m = dict(msg)
            m.pop("prefix", None)
            m.pop("reasoning_content", None)
            clean_msgs.append(m)
        clean_msgs.append({"role": "assistant", "content": assistant_prefix, "prefix": True})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=_sanitize(clean_msgs),
            stop=["</think>"],
        )
        content = response.choices[0].message.content or ""
        self.prefix_usage = response.usage
        if return_usage:
            return content, response.usage
        return content

    async def fim_complete(self, prefix: str, suffix: str = "", max_tokens: int = 256, return_usage: bool = False):
        """FIM completion — fill in the middle between prefix and suffix."""
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/beta")
        kwargs = {
            "model": self.model,
            "prompt": prefix,
            "max_tokens": max_tokens,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if suffix:
            kwargs["suffix"] = suffix
        response = await client.completions.create(**kwargs)
        text = response.choices[0].text or ""
        if return_usage:
            return text, response.usage
        return text
