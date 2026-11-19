"""Cut each stylized rider out of its background (RGBA). Plug in your fine-tuned ISNet as an ONNX file."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from PIL import Image


class Segmenter(Protocol):
    name: str

    def cutout(self, img: Image.Image) -> Image.Image: ...


class RembgSegmenter:
    name = "rembg"

    def __init__(self, model: str = "isnet-general-use"):
        from rembg import new_session  # pip install .[seg]

        self.session = new_session(model)

    def cutout(self, img: Image.Image) -> Image.Image:
        from rembg import remove

        return remove(img.convert("RGB"), session=self.session).convert("RGBA")


class OnnxISNetSegmenter:
    """Generic DIS/ISNet ONNX runner: input 1x3xSxS, normalized (x/255 - 0.5) / 1.0; output mask logits/probs.

    Check that `mean`/`std` match how you exported your fine-tuned model.
    """

    name = "onnx_isnet"

    def __init__(self, onnx_path: str, input_size: int = 1024, mean: float = 0.5, std: float = 1.0, providers: list[str] | None = None):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(onnx_path, providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.size = input_size
        self.mean = mean
        self.std = std

    def cutout(self, img: Image.Image) -> Image.Image:
        rgb = img.convert("RGB")
        x = np.asarray(rgb.resize((self.size, self.size), Image.BILINEAR), dtype=np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = x.transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.input_name: x})[0]
        mask = np.squeeze(out).astype(np.float32)
        lo, hi = float(mask.min()), float(mask.max())
        mask = (mask - lo) / (hi - lo + 1e-8)
        alpha = Image.fromarray((mask * 255).astype(np.uint8)).resize(rgb.size, Image.BILINEAR)
        rgba = rgb.convert("RGBA")
        rgba.putalpha(alpha)
        return rgba


class MockSegmenter:
    """Treats pixels close to the corner color as background. Good enough for synthetic test images."""

    name = "mock"

    def __init__(self, tolerance: int = 40):
        self.tolerance = tolerance

    def cutout(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("RGB")).astype(np.int16)
        bg = arr[-3, 2]  # bottom-left corner (the mock backend writes a label along the top)
        dist = np.abs(arr - bg).sum(axis=2)
        alpha = np.where(dist > self.tolerance, 255, 0).astype(np.uint8)
        rgba = img.convert("RGBA")
        rgba.putalpha(Image.fromarray(alpha))
        return rgba


def build_segmenter(section: dict[str, Any]) -> Segmenter:
    kind = section.get("kind", "mock")
    if kind == "rembg":
        return RembgSegmenter(model=section.get("rembg_model", "isnet-general-use"))
    if kind == "onnx_isnet":
        return OnnxISNetSegmenter(
            onnx_path=section["onnx_path"],
            input_size=int(section.get("input_size", 1024)),
            mean=float(section.get("mean", 0.5)),
            std=float(section.get("std", 1.0)),
        )
    if kind == "mock":
        return MockSegmenter()
    raise ValueError(f"unknown segmenter.kind: {kind}")
