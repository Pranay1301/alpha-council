"""Model adapters for the agent desk: one protocol, any OpenAI-compatible
endpoint (free tiers included), and a deterministic mock for offline runs.

Same philosophy as the rest of the repo: the demo, tests and evals must
run with no API key and no network.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Protocol

FREE_PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "GROQ_API_KEY"),
    "zai": ("https://api.z.ai/api/paas/v4", "glm-4.5-flash", "ZAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "openrouter/free", "OPENROUTER_API_KEY"),
}


@dataclass
class ModelResponse:
    content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Model(Protocol):
    def complete(self, messages: list[dict]) -> ModelResponse:
        ...


class OpenAIModel:
    def __init__(self, model: str, base_url: str | None = None,
                 api_key: str | None = None):
        from openai import OpenAI  # lazy: mock mode needs no deps

        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"],
                             base_url=base_url)
        self.model = model
        self.provider = next((name for name, (url, _, _) in FREE_PROVIDERS.items()
                              if url == base_url), "custom")

    @classmethod
    def from_provider(cls, provider: str, model: str | None = None) -> "OpenAIModel":
        base_url, default_model, key_env = FREE_PROVIDERS[provider]
        return cls(model or default_model, base_url=base_url,
                   api_key=os.environ[key_env])

    def complete(self, messages: list[dict]) -> ModelResponse:
        resp = self.client.chat.completions.create(model=self.model, messages=messages)
        usage = getattr(resp, "usage", None)
        return ModelResponse(
            content=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
        )


class MockModel:
    """Plays back scripted responses in order; extra calls return {}."""

    def __init__(self, script: list[str]):
        self.script = list(script)

    def complete(self, messages: list[dict]) -> ModelResponse:
        if not self.script:
            return ModelResponse(content="{}")
        return ModelResponse(content=self.script.pop(0))


def parse_json(text: str) -> dict:
    """Tolerant JSON extraction for LLM outputs that wrap JSON in prose."""
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}
