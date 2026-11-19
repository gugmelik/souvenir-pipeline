"""Image-edit backend interface. Every backend takes a list of images + a prompt and returns one image."""

from __future__ import annotations

import time
from typing import Protocol

from PIL import Image


class EditBackend(Protocol):
    name: str
    max_images: int

    def edit(
        self,
        images: list[Image.Image],
        prompt: str,
        *,
        negative_prompt: str = " ",
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> Image.Image: ...


class CountingBackend:
    """Wraps a backend to count calls and time them (goes into the run report)."""

    def __init__(self, inner: EditBackend):
        self.inner = inner
        self.name = inner.name
        self.max_images = inner.max_images
        self.calls = 0
        self.seconds = 0.0
        self.dropped_images = 0

    def edit(self, images: list[Image.Image], prompt: str, **kw) -> Image.Image:
        if len(images) > self.max_images:
            self.dropped_images += len(images) - self.max_images
            images = images[: self.max_images]
        t0 = time.perf_counter()
        try:
            return self.inner.edit(images, prompt, **kw)
        finally:
            self.calls += 1
            self.seconds += time.perf_counter() - t0
