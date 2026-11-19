"""Run Qwen-Image-Edit-2511 in-process with diffusers (needs a large GPU; pip install .[local]).

bf16 weights need roughly 40GB+ of VRAM for the transformer alone; use cpu_offload on smaller cards,
or the fp8 genai_instance / Modal workers instead.
"""

from __future__ import annotations

from typing import Any

from PIL import Image


class DiffusersQwenEditBackend:
    name = "diffusers"
    max_images = 3

    def __init__(
        self,
        model_id: str = "Qwen/Qwen-Image-Edit-2511",
        lightning_lora: dict[str, Any] | None = None,
        num_inference_steps: int = 40,
        true_cfg_scale: float = 4.0,
        cpu_offload: bool = True,
    ):
        import torch
        from diffusers import QwenImageEditPlusPipeline

        self.torch = torch
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        if lightning_lora and lightning_lora.get("repo"):
            self.pipe.load_lora_weights(lightning_lora["repo"], weight_name=lightning_lora.get("weight_name"))
            num_inference_steps = int(lightning_lora.get("steps", 4))
            true_cfg_scale = float(lightning_lora.get("true_cfg_scale", 1.0))
        if cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")
        self.steps = num_inference_steps
        self.cfg = true_cfg_scale

    def edit(self, images: list[Image.Image], prompt: str, *, negative_prompt: str = " ", seed: int | None = None,
             width: int | None = None, height: int | None = None) -> Image.Image:
        generator = None
        if seed is not None:
            generator = self.torch.Generator(device="cuda").manual_seed(int(seed))
        with self.torch.inference_mode():
            out = self.pipe(
                image=[im.convert("RGB") for im in images],
                prompt=prompt,
                negative_prompt=negative_prompt or " ",
                true_cfg_scale=self.cfg,
                guidance_scale=1.0,
                num_inference_steps=self.steps,
                num_images_per_prompt=1,
                generator=generator,
            )
        img = out.images[0]
        if width and height:
            img = img.resize((width, height), Image.LANCZOS)
        return img
