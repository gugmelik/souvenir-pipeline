"""Custom nodes: the parts of the occo souvenir pipeline that core ComfyUI has no nodes for.

  Occo Product Spec        load the product JSON (poses, templates, checklist) + optional house-style image
  Occo Detect People       count riders in code, crop body/face per person, optional VLM person cards
  Occo Plan                formation, slots, seeds and every prompt (template or LLM planner)
  Occo Qwen Edit Riders    per-person loop: pose -> stylize with Qwen-Image-Edit (ComfyUI has no loops)
  Occo Cutout              remove each rider's background (rembg if installed, else white-background key)
  Occo Compose             paste riders on the plate at exact positions, one zipline cable each
  Occo Verify              rider count + identity per rider -> text report

Harmonize, effects and the background plate use the core Qwen nodes in the workflow.
"""

from __future__ import annotations

import json
import logging
import math
import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter

import comfy.samplers
import comfy.utils
import folder_paths
import nodes as comfy_nodes

from souvenir.analyzer import analyze
from souvenir.compositor import compose
from souvenir.config import load_product, resolve_asset
from souvenir.detect import MockDetector, build_detector
from souvenir.planner import make_plan
from souvenir.poses import ARMS, draw_pose
from souvenir.schemas import Box
from souvenir.vlm import MockVLM, OpenAICompatibleVLM

log = logging.getLogger("occo_souvenir")
CATEGORY = "occo/souvenir"
_CACHE: dict[str, Any] = {}


# --------------------------------------------------------------------------- tensor <-> PIL


def t2p(t: torch.Tensor, i: int = 0) -> Image.Image:
    arr = (t[i].detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr[:, :, :4] if arr.shape[-1] == 4 else arr[:, :, :3])


def p2t(img: Image.Image, mode: str = "RGB") -> torch.Tensor:
    arr = np.asarray(img.convert(mode)).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None]


def batch(images: list[Image.Image], height: int = 512, bg: tuple[int, int, int] = (128, 128, 128)) -> torch.Tensor:
    """Preview batch: scale every image to the same height and pad to the widest."""
    if not images:
        return torch.zeros(1, 64, 64, 3)
    scaled = [im.convert("RGBA").resize((max(1, round(im.width * height / im.height)), height), Image.LANCZOS) for im in images]
    width = max(s.width for s in scaled)
    out = []
    for s in scaled:
        canvas = Image.new("RGB", (width, height), bg)
        canvas.paste(s, ((width - s.width) // 2, 0), s)
        out.append(p2t(canvas))
    return torch.cat(out, 0)


def _vlm(base_url: str, model: str, api_key: str):
    if not base_url.strip():
        return MockVLM()  # generic wording ("the same outfit"), no network
    key = f"vlm:{base_url}:{model}"
    if key not in _CACHE:
        _CACHE[key] = OpenAICompatibleVLM(base_url=base_url.strip(), api_key=api_key or "EMPTY", model=model)
    return _CACHE[key]


VLM_INPUTS = {
    "vlm_base_url": ("STRING", {"default": "", "tooltip": "OpenAI-compatible endpoint, e.g. http://host:8000/v1 (vLLM + Qwen3-VL). Empty = no VLM."}),
    "vlm_model": ("STRING", {"default": "Qwen/Qwen3-VL-8B-Instruct"}),
    "vlm_api_key": ("STRING", {"default": "EMPTY"}),
}


# --------------------------------------------------------------------------- spec


class OccoProductSpec:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"spec_path": ("STRING", {"default": "configs/products/slotzilla_flight_v1.json",
                                                  "tooltip": "Product spec JSON, absolute or relative to the occo-souvenir-pipeline folder"})},
            "optional": {"style_image": ("IMAGE", {"tooltip": "House-style reference. Without it, stylize uses the spec's text style description."})},
        }

    RETURN_TYPES = ("OCCO_SPEC", "STRING")
    RETURN_NAMES = ("spec", "plate_prompt")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, spec_path: str, style_image: torch.Tensor | None = None):
        spec = load_product(spec_path)
        poses: dict[str, Image.Image] = {}
        for key, path in spec.pose_refs.items():
            p = resolve_asset(path)
            if p.is_file():
                poses[key] = Image.open(p).convert("RGB")
            elif key in ARMS:
                poses[key] = draw_pose(key)
            else:
                raise FileNotFoundError(f"pose reference '{key}' not found at {p}")
        style = t2p(style_image) if style_image is not None else None
        plate_prompt = spec.plate_prompt.replace("in the illustration style of image 1", f"in a {spec.style_description}")
        return ({"spec": spec, "style": style, "poses": poses}, plate_prompt)


# --------------------------------------------------------------------------- detect


class OccoDetectPeople:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detector": (["insightface", "rekognition", "manual"], {"default": "insightface"}),
                "min_face_ratio": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01,
                                             "tooltip": "Faces smaller than this fraction of the largest face are bystanders"}),
                "manual_faces": ("STRING", {"default": "", "multiline": True,
                                            "tooltip": "detector=manual: one face box per line, x0,y0,x1,y1 in pixels"}),
                "rekognition_region": ("STRING", {"default": "us-west-2"}),
            },
            "optional": dict(VLM_INPUTS),
        }

    RETURN_TYPES = ("OCCO_PEOPLE", "INT", "IMAGE", "STRING")
    RETURN_NAMES = ("people", "count", "crops_preview", "person_cards")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, detector, min_face_ratio, manual_faces, rekognition_region,
            vlm_base_url="", vlm_model="", vlm_api_key="EMPTY"):
        img = t2p(image)
        if detector == "manual":
            boxes = []
            for line in manual_faces.strip().splitlines():
                nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", line)]
                if len(nums) == 4:
                    boxes.append(tuple(nums))
            det = MockDetector(boxes=boxes, min_face_ratio=min_face_ratio, det_threshold=0.0)
        else:
            key = f"det:{detector}:{rekognition_region}"
            if key not in _CACHE:
                _CACHE[key] = build_detector({"kind": detector, "region": rekognition_region})
            det = _CACHE[key]
            det.min_face_ratio = min_face_ratio
        out_dir = Path(folder_paths.get_temp_directory()) / f"occo_{uuid.uuid4().hex[:10]}"
        analysis = analyze(img, det, _vlm(vlm_base_url, vlm_model, vlm_api_key), out_dir, source="comfyui")
        crops = []
        for p in analysis.persons:
            crops.append(Image.open(analysis.crops[str(p.id)]["body"]))
        cards = json.dumps([c.model_dump(exclude={"card_ok"}) for c in analysis.cards], indent=1)
        info = f"{analysis.n_people} rider(s), {analysis.ignored_faces} bystander face(s) ignored\n{cards}"
        return ({"analysis": analysis, "dir": str(out_dir)}, analysis.n_people, batch(crops), info)


# --------------------------------------------------------------------------- plan


class OccoPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "spec": ("OCCO_SPEC",),
                "people": ("OCCO_PEOPLE",),
                "seed": ("INT", {"default": 7, "min": 0, "max": 2**31 - 1, "control_after_generate": True}),
                "planner": (["template", "llm"], {"default": "template", "tooltip": "llm needs the VLM endpoint below"}),
                "max_prompt_tokens": ("INT", {"default": 120, "min": 40, "max": 1000}),
            },
            "optional": dict(VLM_INPUTS),
        }

    RETURN_TYPES = ("OCCO_PLAN", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("plan", "plan_text", "harmonize_prompt", "effects_prompt")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, spec, people, seed, planner, max_prompt_tokens, vlm_base_url="", vlm_model="", vlm_api_key="EMPTY"):
        analysis = people["analysis"]
        if analysis.n_people == 0:
            raise ValueError("Occo Plan: no riders detected (check the detector or use detector=manual)")
        s = spec["spec"]
        if analysis.n_people > s.max_people:
            raise ValueError(f"Occo Plan: {analysis.n_people} riders, product supports up to {s.max_people}")
        plan = make_plan(s, analysis, _vlm(vlm_base_url, vlm_model, vlm_api_key), mode=planner, seed=seed,
                         max_prompt_tokens=max_prompt_tokens)
        lines = [f"formation: {plan.formation}   riders: {plan.n_people}   planner: {plan.planner}"]
        for pp in plan.per_person:
            lines.append(f"\nperson {pp.id}  pose ref: {pp.pose_ref}")
            for st in pp.steps[:2]:
                lines.append(f"  {st.op} (seed {st.seed}): {st.prompt}")
        lines += [f"\nharmonize: {plan.harmonize.prompt}", f"effects: {plan.effects.prompt}"]
        lines += [f"warning: {w}" for w in plan.warnings]
        return (plan, "\n".join(lines), plan.harmonize.prompt, plan.effects.prompt)


# --------------------------------------------------------------------------- Qwen edit loop


def _unwrap(out):
    if hasattr(out, "result"):
        return out.result[0]
    return out[0]


def _node(name: str):
    cls = comfy_nodes.NODE_CLASS_MAPPINGS[name]
    if hasattr(cls, "execute"):  # v3 node API
        return lambda **kw: _unwrap(cls.execute(**kw))
    inst = cls()
    return lambda **kw: _unwrap(getattr(inst, cls.FUNCTION)(**kw))


def _scale_to_pixels(image: torch.Tensor, megapixels: float = 1.0, multiple: int = 16) -> torch.Tensor:
    samples = image.movedim(-1, 1)
    scale = math.sqrt(megapixels * 1024 * 1024 / (samples.shape[2] * samples.shape[3]))
    w = max(multiple, round(samples.shape[3] * scale / multiple) * multiple)
    h = max(multiple, round(samples.shape[2] * scale / multiple) * multiple)
    return comfy.utils.common_upscale(samples, w, h, "lanczos", "disabled").movedim(1, -1)


def qwen_edit(model, clip, vae, images: list[torch.Tensor], prompt: str, seed: int, steps: int, cfg: float,
              sampler_name: str, scheduler: str, reference_method: str) -> torch.Tensor:
    """Same math as the core graph: TextEncodeQwenImageEditPlus -> reference method -> KSampler -> VAEDecode."""
    encode = _node("TextEncodeQwenImageEditPlus")
    method = _node("FluxKontextMultiReferenceLatentMethod")
    base = _scale_to_pixels(images[0][:, :, :, :3])
    refs = {"image1": base}
    for i, im in enumerate(images[1:3], start=2):
        refs[f"image{i}"] = im[:, :, :, :3]
    pos = method(conditioning=encode(clip=clip, prompt=prompt, vae=vae, **refs), reference_latents_method=reference_method)
    neg = method(conditioning=encode(clip=clip, prompt="", vae=vae, **refs), reference_latents_method=reference_method)
    latent = {"samples": vae.encode(base)}
    out = comfy_nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, pos, neg, latent, denoise=1.0)[0]
    decoded = vae.decode(out["samples"])
    if decoded.ndim == 5:
        decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
    return decoded


def _without_style_image(prompt: str, style_description: str) -> str:
    style = style_description or "3D animated feature-film style"
    p = re.sub(r"in the (illustration )?style of image 2", f"in a {style}", prompt)
    return p.replace("image 3", "image 2")


class OccoQwenEditRiders:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Qwen-Image-Edit-2511 after AuraFlow shift, CFGNorm and the Lightning LoRA"}),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "spec": ("OCCO_SPEC",),
                "people": ("OCCO_PEOPLE",),
                "plan": ("OCCO_PLAN",),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "reference_method": (["index_timestep_zero", "index", "offset"], {"default": "index_timestep_zero"}),
            }
        }

    RETURN_TYPES = ("OCCO_RIDERS", "IMAGE", "IMAGE")
    RETURN_NAMES = ("riders", "posed_preview", "styled_preview")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, model, clip, vae, spec, people, plan, steps, cfg, sampler_name, scheduler, reference_method):
        analysis = people["analysis"]
        s = spec["spec"]
        style = spec["style"]
        pbar = comfy.utils.ProgressBar(2 * len(plan.per_person))
        riders, posed_imgs, styled_imgs = [], [], []
        for pp in plan.per_person:
            crops = analysis.crops[str(pp.id)]
            body = p2t(Image.open(crops["body"]))
            face = p2t(Image.open(crops["face"]))
            pose_ref = p2t(spec["poses"][pp.pose_ref])
            pose_step, style_step = pp.steps[0], pp.steps[1]

            posed = qwen_edit(model, clip, vae, [body, pose_ref], pose_step.prompt, pose_step.seed or 0,
                              steps, cfg, sampler_name, scheduler, reference_method)
            pbar.update(1)
            if style is not None:
                imgs, prompt = [posed, p2t(style), face], style_step.prompt
            else:
                imgs, prompt = [posed, face], _without_style_image(style_step.prompt, s.style_description)
            styled = qwen_edit(model, clip, vae, imgs, prompt, style_step.seed or 0,
                               steps, cfg, sampler_name, scheduler, reference_method)
            pbar.update(1)
            riders.append({"id": pp.id, "posed": t2p(posed), "styled": t2p(styled)})
            posed_imgs.append(riders[-1]["posed"])
            styled_imgs.append(riders[-1]["styled"])
        return (riders, batch(posed_imgs), batch(styled_imgs))


# --------------------------------------------------------------------------- cutout


def _white_key(img: Image.Image, tolerance: int = 38) -> Image.Image:
    """Remove the light, plain background that the stylize prompt asks for: flood-fill from the borders."""
    from scipy import ndimage

    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    bg = np.median(border, axis=0)
    close = np.abs(arr - bg).sum(axis=2) <= tolerance * 3
    labels, _ = ndimage.label(close)
    edge_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    background = np.isin(labels, edge_labels[edge_labels > 0])
    alpha = np.where(background, 0, 255).astype(np.uint8)
    alpha = np.asarray(Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(1.2)))
    out = img.convert("RGBA")
    out.putalpha(Image.fromarray(alpha))
    return out


class OccoCutout:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "riders": ("OCCO_RIDERS",),
            "method": (["auto", "rembg", "white_key"], {"default": "auto",
                       "tooltip": "auto = rembg (isnet-general-use) if installed, else white-background key"}),
            "white_tolerance": ("INT", {"default": 38, "min": 1, "max": 120}),
        }}

    RETURN_TYPES = ("OCCO_RIDERS", "IMAGE", "MASK")
    RETURN_NAMES = ("riders", "cutouts_preview", "masks")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, riders, method, white_tolerance):
        use_rembg = method == "rembg"
        if method == "auto":
            try:
                import rembg  # noqa: F401
                use_rembg = True
            except ImportError:
                use_rembg = False
        out, previews, masks = [], [], []
        for r in riders:
            if use_rembg:
                from rembg import new_session, remove

                if "rembg" not in _CACHE:
                    _CACHE["rembg"] = new_session("isnet-general-use")
                rgba = remove(r["styled"], session=_CACHE["rembg"]).convert("RGBA")
            else:
                rgba = _white_key(r["styled"], white_tolerance)
            out.append({**r, "cutout": rgba})
            previews.append(rgba)
        prev = batch(previews)
        h = prev.shape[1]
        for rgba in previews:
            a = rgba.getchannel("A").resize((max(1, round(rgba.width * h / rgba.height)), h))
            pad = Image.new("L", (prev.shape[2], h), 0)
            pad.paste(a, ((prev.shape[2] - a.width) // 2, 0))
            masks.append(torch.from_numpy(np.asarray(pad).astype(np.float32) / 255.0)[None])
        return (out, prev, torch.cat(masks, 0))


# --------------------------------------------------------------------------- compose


class OccoCompose:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "spec": ("OCCO_SPEC",), "plan": ("OCCO_PLAN",), "riders": ("OCCO_RIDERS",), "plate": ("IMAGE",),
            "canvas": (["from plate", "from spec"], {"default": "from plate",
                       "tooltip": "from plate: keep the plate's aspect ratio (long side = long_side). from spec: the spec's canvas, center-cropped."}),
            "long_side": ("INT", {"default": 1600, "min": 512, "max": 4096, "step": 16}),
        }}

    RETURN_TYPES = ("IMAGE", "OCCO_LAYOUT")
    RETURN_NAMES = ("composite", "layout")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, spec, plan, riders, plate, canvas="from plate", long_side=1600):
        from souvenir.schemas import Canvas

        s = spec["spec"]
        size = s.canvas
        if canvas == "from plate":
            ph, pw = int(plate.shape[1]), int(plate.shape[2])
            k = long_side / max(pw, ph)
            size = Canvas(width=max(16, round(pw * k / 16) * 16), height=max(16, round(ph * k / 16) * 16))
        cutouts = {r["id"]: r["cutout"] for r in riders}
        missing = [pid for pid in plan.slots if pid not in cutouts]
        if missing:
            raise ValueError(f"Occo Compose: riders {missing} have no cutout (connect Occo Cutout first)")
        img, boxes = compose(t2p(plate), cutouts, plan.slots, size, s.cable_direction, s.cable_angle_deg)
        return (p2t(img), {"boxes": boxes, "canvas": (size.width, size.height)})


# --------------------------------------------------------------------------- verify


class OccoVerify:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "spec": ("OCCO_SPEC",),
                "people": ("OCCO_PEOPLE",),
                "layout": ("OCCO_LAYOUT",),
                "identity": (["insightface", "vlm_judge", "none"], {"default": "insightface"}),
                "identity_threshold": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01,
                                                 "tooltip": "Calibrate per style; photo-trained face models score cartoons low"}),
            },
            "optional": dict(VLM_INPUTS),
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, spec, people, layout, identity, identity_threshold, vlm_base_url="", vlm_model="", vlm_api_key="EMPTY"):
        from souvenir.identity import InsightFaceScorer, VLMJudgeScorer
        from souvenir.ocr import NoOCR
        from souvenir.verifier import Verifier

        vlm = _vlm(vlm_base_url, vlm_model, vlm_api_key) if vlm_base_url.strip() else _NoVLM()
        try:
            detector = _CACHE.setdefault("det:insightface:us-west-2", build_detector({"kind": "insightface"}))
        except Exception:
            detector = MockDetector()
        if identity == "insightface":
            scorer = _CACHE.setdefault("score:insightface", InsightFaceScorer())
        elif identity == "vlm_judge":
            scorer = VLMJudgeScorer(vlm)
        else:
            scorer = _NoScorer()
        s = spec["spec"]
        v = Verifier(s, vlm, scorer, detector, NoOCR(), identity_threshold=identity_threshold)
        boxes: dict[int, Box] = layout["boxes"]
        img = t2p(image).resize(layout["canvas"], Image.LANCZOS)
        rep = v.verify_composite("final", img, people["analysis"], boxes)
        undecided = sum(1 for c in rep.checks if c.passed is None)
        if any(c.passed is False for c in rep.checks):
            overall = "FAIL"
        elif undecided:
            overall = f"NO FAILURES, but {undecided} check(s) could not be decided (no VLM / no readable face)"
        else:
            overall = "PASS"
        lines = [f"overall: {overall}"]
        for c in rep.checks:
            mark = "ok  " if c.passed else "FAIL" if c.passed is False else "?   "
            lines.append(f"{mark} {c.name}: {c.detail}")
        return ("\n".join(lines),)


class _NoVLM:
    """No VLM configured: checklist/count questions come back 'unknown' instead of a fake 'yes'."""

    name = "none"

    def chat_json(self, *a, **k):
        raise RuntimeError("no VLM configured")


class _NoScorer:
    name = "off"

    def score(self, *a, **k):
        return None


NODE_CLASS_MAPPINGS = {
    "OccoProductSpec": OccoProductSpec,
    "OccoDetectPeople": OccoDetectPeople,
    "OccoPlan": OccoPlan,
    "OccoQwenEditRiders": OccoQwenEditRiders,
    "OccoCutout": OccoCutout,
    "OccoCompose": OccoCompose,
    "OccoVerify": OccoVerify,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OccoProductSpec": "Occo Product Spec",
    "OccoDetectPeople": "Occo Detect People",
    "OccoPlan": "Occo Plan",
    "OccoQwenEditRiders": "Occo Qwen Edit Riders (pose + stylize loop)",
    "OccoCutout": "Occo Cutout",
    "OccoCompose": "Occo Compose (exact layout)",
    "OccoVerify": "Occo Verify",
}
