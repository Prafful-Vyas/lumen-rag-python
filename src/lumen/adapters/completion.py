from __future__ import annotations

import math
import re
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from lumen.adapters.pricing import estimate_completion_cost_usd
from lumen.ports import ChatMessage, CompletionUsage, DoneEvent, TokenEvent


class OpenAICompletionProvider:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def complete(
        self, messages: list[ChatMessage], max_tokens: int | None = None, temperature: float = 0.2
    ) -> tuple[str, CompletionUsage]:
        res = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        usage = CompletionUsage(
            prompt_tokens=res.usage.prompt_tokens if res.usage else 0,
            completion_tokens=res.usage.completion_tokens if res.usage else 0,
        )
        return (res.choices[0].message.content or "") if res.choices else "", usage

    async def stream_complete(
        self, messages: list[ChatMessage], max_tokens: int | None = None, temperature: float = 0.2
    ) -> AsyncIterator[TokenEvent | DoneEvent]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        usage = CompletionUsage()
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield TokenEvent(chunk.choices[0].delta.content)
            if chunk.usage:
                usage = CompletionUsage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
        yield DoneEvent(usage)

    def estimate_cost_usd(self, usage: CompletionUsage) -> float:
        return estimate_completion_cost_usd(self.model, usage.prompt_tokens, usage.completion_tokens)


class MockCompletionProvider:
    """Deterministic offline stand-in for the generation LLM. It finds the source markers present
    in the assembled prompt and writes a short answer citing the first two, which is enough to
    exercise streaming, citation parsing and validation without an API key."""

    model = "mock-completion"

    def _answer(self, messages: list[ChatMessage]) -> str:
        user = "\n".join(m.content for m in messages if m.role == "user")
        markers = [int(m) for m in re.findall(r"^\[(\d+)\]", user, flags=re.M)]
        match = re.search(r"Question:\s*([\s\S]+)$", user)
        question = match.group(1).strip() if match else "your question"
        if not markers:
            return "I couldn't find anything in your documents about this."
        cited = " ".join(f"[{m}]" for m in markers[:2])
        return (
            "[MOCK ANSWER - set OPENAI_API_KEY for a real one] Based on the retrieved sources, "
            f'here is a grounded answer to "{question}" {cited}.'
        )

    @staticmethod
    def _usage(messages: list[ChatMessage], text: str) -> CompletionUsage:
        return CompletionUsage(math.ceil(sum(len(m.content) for m in messages) / 4), math.ceil(len(text) / 4))

    async def complete(
        self, messages: list[ChatMessage], max_tokens: int | None = None, temperature: float = 0.2
    ) -> tuple[str, CompletionUsage]:
        text = self._answer(messages)
        return text, self._usage(messages, text)

    async def stream_complete(
        self, messages: list[ChatMessage], max_tokens: int | None = None, temperature: float = 0.2
    ) -> AsyncIterator[TokenEvent | DoneEvent]:
        text = self._answer(messages)
        for word in re.split(r"(?<= )", text):
            if word:
                yield TokenEvent(word)
        yield DoneEvent(self._usage(messages, text))

    def estimate_cost_usd(self, usage: CompletionUsage) -> float:
        return 0.0
