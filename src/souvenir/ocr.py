"""OCR for text checks: banned brand text must be absent, required text ("Las Vegas 2026") present."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from PIL import Image


class OCR(Protocol):
    name: str

    def read(self, img: Image.Image) -> list[str]: ...


class EasyOCR:
    name = "easyocr"

    def __init__(self, langs: list[str] | None = None, gpu: bool = False):
        import easyocr  # pip install .[ocr]

        self.reader = easyocr.Reader(langs or ["en"], gpu=gpu)

    def read(self, img: Image.Image) -> list[str]:
        return [str(t) for t in self.reader.readtext(np.asarray(img.convert("RGB")), detail=0)]


class NoOCR:
    name = "none"

    def read(self, img: Image.Image) -> list[str]:
        return []


def build_ocr(section: dict[str, Any]) -> OCR:
    if not section.get("enabled", False):
        return NoOCR()
    return EasyOCR(langs=section.get("langs", ["en"]), gpu=bool(section.get("gpu", False)))


def normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())
