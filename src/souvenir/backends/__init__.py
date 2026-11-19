"""Backend factory. Pick one in configs/pipeline.yaml -> backend.kind."""

from __future__ import annotations

from typing import Any

from souvenir.backends.base import CountingBackend, EditBackend


def build_backend(kind: str, section: dict[str, Any]) -> EditBackend:
    if kind == "mock":
        from souvenir.backends.mock import MockEditBackend

        return MockEditBackend()
    if kind == "gpu_instances":
        from souvenir.backends.gpu_instances import build

        return build(section)
    if kind == "genai_instance":
        from souvenir.backends.genai_instance import GenaiInstanceBackend

        return GenaiInstanceBackend(
            url=section.get("url"),
            deployments_file=section.get("deployments_file"),
            model=section.get("model", "qwen-image-edit-2511"),
            num_inference_steps=int(section.get("num_inference_steps", 40)),
            true_cfg_scale=float(section.get("true_cfg_scale", 4.0)),
            timeout_s=int(section.get("timeout_s", 600)),
        )
    if kind == "modal_comfy":
        from souvenir.backends.modal_comfy import ModalComfyBackend, S3Uploader

        uploader = S3Uploader(
            bucket=section["s3_bucket"],
            prefix=section.get("s3_prefix", "souvenir-tmp/"),
            region=section.get("region", "us-west-2"),
        )
        return ModalComfyBackend(
            url=section.get("url"),
            uploader=uploader,
            width=int(section.get("width", 1200)),
            height=int(section.get("height", 1800)),
        )
    if kind == "diffusers":
        from souvenir.backends.diffusers_local import DiffusersQwenEditBackend

        return DiffusersQwenEditBackend(
            model_id=section.get("model_id", "Qwen/Qwen-Image-Edit-2511"),
            lightning_lora=section.get("lightning_lora"),
            num_inference_steps=int(section.get("num_inference_steps", 40)),
            true_cfg_scale=float(section.get("true_cfg_scale", 4.0)),
            cpu_offload=bool(section.get("cpu_offload", True)),
        )
    if kind == "gemini":
        from souvenir.backends.gemini import GeminiImageBackend

        return GeminiImageBackend(model=section.get("model", "gemini-2.5-flash-image"),
                                  api_key_env=section.get("api_key_env", "GEMINI_API_KEY"))
    raise ValueError(f"unknown backend kind: {kind}")


__all__ = ["build_backend", "CountingBackend", "EditBackend"]
