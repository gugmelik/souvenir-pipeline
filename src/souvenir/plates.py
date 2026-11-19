"""Draft background-plate candidates (scene with NO people) for human approval.

Plates are made once per product, reviewed by a person (and the client), then copied to
assets/plates/approved/. Text like "Las Vegas 2026" and brand-safety are settled here, once,
instead of in every guest's prompt.
"""

from __future__ import annotations

from pathlib import Path

from souvenir.backends.base import EditBackend
from souvenir.imaging import load_image, save
from souvenir.ocr import OCR, normalize
from souvenir.schemas import ProductSpec


def generate_plates(spec: ProductSpec, backend: EditBackend, ocr: OCR, out_dir: Path, n: int = 6, seed: int = 0) -> list[dict]:
    if not spec.plate_prompt:
        raise ValueError("product spec has no plate_prompt")
    if not spec.style_refs:
        raise ValueError("product spec has no style_refs (the plate is drawn in the house style)")
    style = load_image(spec.style_refs[0])
    W, H = spec.canvas.width, spec.canvas.height
    results = []
    for i in range(n):
        img = backend.edit([style], spec.plate_prompt, seed=seed + i, width=W, height=H)
        texts = ocr.read(img)
        joined = normalize(" ".join(texts))
        banned = [b for b in spec.banned_text_ocr if normalize(b) in joined]
        prefix = "REJECT_" if banned else ""
        path = save(img, out_dir / f"{prefix}{spec.product}_plate_{i:02d}.png")
        results.append({"path": path, "ocr": texts, "banned": banned})
    return results
