"""VLM / LLM client used by the analyzer, planner and verifier.

`OpenAICompatibleVLM` talks to any OpenAI-compatible chat endpoint:
  * vLLM serving Qwen3-VL on your own GPU (recommended: guest faces stay in your infrastructure)
  * OpenRouter / OpenAI / Gemini's OpenAI-compatible endpoint (fine for planning text;
    check the provider's data terms before sending guest photos)
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from PIL import Image

from souvenir.imaging import limit_size, to_data_uri


class VLMError(RuntimeError):
    pass


def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object in a model reply (tolerates ``` fences and chatter)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start < 0:
        raise VLMError(f"no JSON object in reply: {text[:200]}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise VLMError(f"unterminated JSON in reply: {text[:200]}")


class VLM(Protocol):
    name: str

    def chat_json(self, system: str, user: str, images: list[Image.Image] | None = None) -> dict[str, Any]: ...


class OpenAICompatibleVLM:
    def __init__(
        self,
        base_url: str | None,
        api_key: str | None,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        image_max_side: int = 1024,
        json_mode: bool = True,
        timeout_s: float = 120,
    ):
        from openai import OpenAI  # optional dependency: pip install .[vlm]

        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout_s)
        self.model = model
        self.name = f"openai:{model}"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_max_side = image_max_side
        self.json_mode = json_mode

    def chat_json(self, system: str, user: str, images: list[Image.Image] | None = None) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for img in images or []:
            uri = to_data_uri(limit_size(img.convert("RGB"), self.image_max_side), fmt="jpeg")
            content.append({"type": "image_url", "image_url": {"url": uri}})
        content.append({"type": "text", "text": user})
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # some servers reject response_format; retry without it once
            if self.json_mode:
                kwargs.pop("response_format", None)
                resp = self.client.chat.completions.create(**kwargs)
            else:
                raise VLMError(str(exc)) from exc
        text = resp.choices[0].message.content or ""
        return extract_json(text)


class MockVLM:
    """Deterministic stand-in so the whole pipeline runs offline (tests, demo)."""

    name = "mock"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def chat_json(self, system: str, user: str, images: list[Image.Image] | None = None) -> dict[str, Any]:
        task = _task_of(system)
        self.calls.append(task)
        if task == "person_card":
            # Empty descriptions on purpose: prompts then say "the same outfit" instead of inventing clothes.
            return {"clothing": "", "hair": "", "accessories": [], "age_group": "unknown", "body_type": ""}
        if task == "checklist":
            questions = re.findall(r"^\d+\.\s*(.+)$", user, re.M)
            return {"answers": [{"q": q, "answer": "yes", "confidence": 0.9} for q in questions]}
        if task == "count":
            return {"count": None}
        if task == "identity":
            return {"same_person": True, "confidence": 0.8}
        if task == "planner":
            raise VLMError("mock VLM does not plan; planner falls back to the template plan")
        return {}


def _task_of(system: str) -> str:
    m = re.search(r"TASK:\s*([a-z_]+)", system)
    return m.group(1) if m else "unknown"


def build_vlm(section: dict[str, Any]) -> VLM:
    kind = section.get("kind", "mock")
    if kind == "mock":
        return MockVLM()
    if kind == "openai":
        return OpenAICompatibleVLM(
            base_url=section.get("base_url"),
            api_key=section.get("api_key"),
            model=section.get("model", "Qwen/Qwen3-VL-8B-Instruct"),
            temperature=float(section.get("temperature", 0.2)),
            max_tokens=int(section.get("max_tokens", 1500)),
            image_max_side=int(section.get("image_max_side", 1024)),
            json_mode=bool(section.get("json_mode", True)),
        )
    raise ValueError(f"unknown vlm.kind: {kind}")
