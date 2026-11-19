"""Backend for your Modal ComfyUI app (C:\\dev\\modal\\modal_comfy.py, Qwen-Image-Edit-2511 fp8 + Lightning 4-step).

That endpoint takes two image URLs: `input_image_url` (node 197, resized to width x height)
and `background_image_url` (node 198). So this backend:
  * uploads each image to S3 and passes presigned URLs,
  * supports at most 2 images (extra images are dropped by CountingBackend; prefer genai_instance
    for 3-image steps such as stylize = [posed person, style ref, face crop]),
  * repeats image 1 when a step has only one image.
"""

from __future__ import annotations

import base64
import io
import uuid

import requests
from PIL import Image

from souvenir.imaging import limit_size, to_png_bytes


class S3Uploader:
    def __init__(self, bucket: str, prefix: str = "souvenir-tmp/", region: str = "us-west-2", expires_s: int = 3600):
        import boto3  # pip install .[aws]

        self.s3 = boto3.client("s3", region_name=region)
        self.bucket = bucket
        self.prefix = prefix
        self.expires_s = expires_s

    def put(self, img: Image.Image) -> str:
        key = f"{self.prefix}{uuid.uuid4().hex}.png"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=to_png_bytes(img), ContentType="image/png")
        return self.s3.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=self.expires_s
        )


class ModalComfyBackend:
    name = "modal_comfy"
    max_images = 2

    def __init__(self, url: str, uploader: S3Uploader, width: int = 1200, height: int = 1800, timeout_s: int = 1200):
        if not url:
            raise ValueError("modal_comfy backend needs `url` (printed by `modal deploy modal_comfy.py`)")
        self.url = url
        self.uploader = uploader
        self.width = width
        self.height = height
        self.timeout_s = timeout_s

    def edit(self, images: list[Image.Image], prompt: str, *, negative_prompt: str = " ", seed: int | None = None,
             width: int | None = None, height: int | None = None) -> Image.Image:
        imgs = [limit_size(im.convert("RGB"), 1536) for im in images[:2]]
        if len(imgs) == 1:
            imgs.append(imgs[0])
        payload = {
            "input_image_url": self.uploader.put(imgs[0]),
            "background_image_url": self.uploader.put(imgs[1]),
            "prompt": prompt,
            "negative_prompt": "" if negative_prompt == " " else negative_prompt,
            "width": width or self.width,
            "height": height or self.height,
        }
        if seed is not None:
            payload["seed"] = int(seed) % 2147483647
        r = requests.post(self.url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        if not data.get("images"):
            raise RuntimeError(f"modal endpoint returned no images: {str(data)[:300]}")
        raw = base64.b64decode(data["images"][0]["image_base64"])
        return Image.open(io.BytesIO(raw)).convert("RGB")
