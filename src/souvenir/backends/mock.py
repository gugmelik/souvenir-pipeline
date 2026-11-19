"""Offline stand-in for an image model: returns image 1 slightly changed and labeled with the prompt."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageOps


class MockEditBackend:
    name = "mock"
    max_images = 4

    def edit(self, images: list[Image.Image], prompt: str, *, negative_prompt: str = " ", seed: int | None = None,
             width: int | None = None, height: int | None = None) -> Image.Image:
        base = images[0].convert("RGB").copy()
        if width and height:
            base = ImageOps.fit(base, (width, height))
        draw = ImageDraw.Draw(base)
        label = prompt[:60] + ("..." if len(prompt) > 60 else "")
        draw.rectangle((0, 0, base.width, 14), fill=(255, 255, 255))
        draw.text((3, 1), label, fill=(200, 0, 120))
        return base
