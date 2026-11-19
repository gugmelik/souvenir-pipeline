"""Small PIL helpers: loading from paths/URLs, base64, crops and resizing."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from souvenir.schemas import Box


def load_image(src: str | Path | Image.Image, mode: str = "RGB") -> Image.Image:
    if isinstance(src, Image.Image):
        return src.convert(mode)
    s = str(src)
    if s.startswith("http://") or s.startswith("https://"):
        import requests

        r = requests.get(s, timeout=60)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert(mode)
    if s.startswith("data:"):
        return Image.open(io.BytesIO(base64.b64decode(s.split(",", 1)[1]))).convert(mode)
    return Image.open(s).convert(mode)


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def to_jpeg_bytes(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def to_data_uri(img: Image.Image, fmt: str = "png") -> str:
    data = to_png_bytes(img) if fmt == "png" else to_jpeg_bytes(img)
    mime = "image/png" if fmt == "png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def from_b64(value: str) -> Image.Image:
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")


def crop(img: Image.Image, box: Box) -> Image.Image:
    b = box.clip(img.width, img.height)
    return img.crop(b.as_int_tuple())


def limit_size(img: Image.Image, max_side: int = 1536) -> Image.Image:
    """Downscale so the longest side is <= max_side (keeps aspect)."""
    scale = max_side / max(img.size)
    if scale >= 1:
        return img
    return img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)


def fit_cover(img: Image.Image, width: int, height: int) -> Image.Image:
    """Resize + center-crop to exactly width x height."""
    scale = max(width / img.width, height / img.height)
    resized = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def trim_alpha(rgba: Image.Image, threshold: int = 8) -> Image.Image:
    """Crop an RGBA image to the bounding box of its visible pixels."""
    alpha = np.asarray(rgba.getchannel("A"))
    ys, xs = np.where(alpha > threshold)
    if len(xs) == 0:
        return rgba
    return rgba.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def save(img: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return str(path)
