"""Backend for your existing C:\\dev\\genai_instance worker (RunPod pod, diffusers QwenImageEditPlusPipeline).

Request:  POST {url}/invoke  {"prompt", "image": [data URIs], "negative_prompt", "true_cfg_scale",
                              "num_inference_steps", "seed", "output_format"}
Response: {"model", "output_format", "images": [base64, ...]}

The URL comes from `url` in pipeline.yaml, or from genai_instance's .genai/deployments.json
(written by `scripts\\start.bat qwen-image-edit-2511`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from souvenir.imaging import from_b64, limit_size, to_data_uri


def url_from_deployments(path: str | Path, model: str, worker: str | None = None) -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    workers = (data.get(model) or {}).get("workers") or {}
    record = workers.get(worker or model) or next(iter(workers.values()), None)
    if not record or not record.get("url"):
        raise RuntimeError(f"no running deployment for {model} in {path}; start one with genai_instance first")
    return str(record["url"]).rstrip("/")


class GenaiInstanceBackend:
    name = "genai_instance"
    max_images = 4  # catalog guard in genai_instance/models/qwen-image-edit-2511.yaml

    def __init__(
        self,
        url: str | None = None,
        deployments_file: str | None = None,
        model: str = "qwen-image-edit-2511",
        num_inference_steps: int = 40,
        true_cfg_scale: float = 4.0,
        timeout_s: int = 600,
        image_max_side: int = 1536,
    ):
        if not url:
            if not deployments_file:
                raise ValueError("genai_instance backend needs `url` or `deployments_file`")
            url = url_from_deployments(deployments_file, model)
        self.url = url.rstrip("/")
        self.steps = num_inference_steps
        self.cfg = true_cfg_scale
        self.timeout_s = timeout_s
        self.image_max_side = image_max_side

    def edit(self, images: list[Image.Image], prompt: str, *, negative_prompt: str = " ", seed: int | None = None,
             width: int | None = None, height: int | None = None) -> Image.Image:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "image": [to_data_uri(limit_size(im.convert("RGB"), self.image_max_side)) for im in images],
            "negative_prompt": negative_prompt or " ",
            "true_cfg_scale": self.cfg,
            "num_inference_steps": self.steps,
            "output_format": "png",
        }
        if seed is not None:
            payload["seed"] = int(seed) % 2147483647
        r = requests.post(f"{self.url}/invoke", json=payload, timeout=self.timeout_s)
        if r.status_code >= 400:
            raise RuntimeError(f"genai_instance returned {r.status_code}: {r.text[:500]}")
        data = r.json()
        if not data.get("images"):
            raise RuntimeError(f"genai_instance returned no images: {str(data)[:300]}")
        out = from_b64(data["images"][0])
        if width and height:
            out = out.resize((width, height), Image.LANCZOS)
        return out
