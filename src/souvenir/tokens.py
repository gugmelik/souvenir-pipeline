"""Prompt token counting.

Qwen-Image-Edit encodes prompts with a Qwen2.5-VL text encoder. Diffusers defaults to
max_sequence_length=512 (hard limit 1024); Alibaba's API truncates older edit models at 800 tokens.
Anything past the limit is silently dropped, so the END of a long brief is what gets lost.
"""

from __future__ import annotations

import functools
import math
import re

QWEN_LIMITS = {"diffusers_default": 512, "diffusers_max": 1024, "alibaba_api": 800}


@functools.lru_cache(maxsize=4)
def _tokenizer(name: str):
    from transformers import AutoTokenizer  # optional dependency

    return AutoTokenizer.from_pretrained(name)


def count_tokens(text: str, tokenizer: str | None = None) -> int:
    """Exact count with a HF tokenizer if available, otherwise a conservative estimate."""
    if tokenizer:
        try:
            return len(_tokenizer(tokenizer)(text)["input_ids"])
        except Exception:
            pass
    # Heuristic: ~1.3 tokens per word-ish piece for English prose with markdown.
    pieces = re.findall(r"\w+|[^\w\s]", text)
    return math.ceil(len(pieces) * 1.05 + len(text) / 40)


def budget_report(text: str, tokenizer: str | None = None) -> dict[str, object]:
    n = count_tokens(text, tokenizer)
    report: dict[str, object] = {"tokens": n, "exact": bool(tokenizer)}
    for name, limit in QWEN_LIMITS.items():
        report[f"over_{name}"] = n > limit
    if n > QWEN_LIMITS["diffusers_default"]:
        # roughly where the default limit cuts the text
        words = text.split()
        cut = int(len(words) * QWEN_LIMITS["diffusers_default"] / n)
        report["first_words_lost_at_512"] = " ".join(words[cut : cut + 12])
    return report
