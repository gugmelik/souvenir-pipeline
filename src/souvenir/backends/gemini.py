"""Nano Banana (Gemini image) backend - for the BASELINE comparison only.

Gemini API terms say you may not use the service to develop models that compete with it, so do not
train on these outputs without a legal review. Using it to score/compare is the intended use here.
"""

from __future__ import annotations

import io
import os

from PIL import Image


class GeminiImageBackend:
    name = "gemini"
    max_images = 3

    def __init__(self, model: str = "gemini-2.5-flash-image", api_key_env: str = "GEMINI_API_KEY"):
        from google import genai  # pip install .[gemini]

        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError(f"set {api_key_env} to use the Gemini baseline")
        self.client = genai.Client(api_key=key)
        self.model = model

    def edit(self, images: list[Image.Image], prompt: str, *, negative_prompt: str = " ", seed: int | None = None,
             width: int | None = None, height: int | None = None) -> Image.Image:
        resp = self.client.models.generate_content(model=self.model, contents=[prompt, *images])
        for cand in resp.candidates or []:
            for part in cand.content.parts or []:
                data = getattr(part, "inline_data", None)
                if data is not None and data.data:
                    img = Image.open(io.BytesIO(data.data)).convert("RGB")
                    if width and height:
                        img = img.resize((width, height), Image.LANCZOS)
                    return img
        raise RuntimeError("Gemini returned no image (possibly blocked by safety filters)")
