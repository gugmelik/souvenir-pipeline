"""Export the pipeline's plan for one photo as a single ComfyUI API-format workflow (preview before deploying).

Nodes and model files are the same as C:\\dev\\occo\\comfyui-api\\workflows\\qwen_image_edit_2511.json
(Qwen-Image-Edit-2511 fp8mixed + Lightning 4-step, core nodes only), so the workflow runs on your Vast
ComfyUI endpoint (`scripts\\run.py ... --workflow-json`) or in any ComfyUI with those models.

Stages, each with its own SaveImage so you can look at every step:
  plate        approved plate (LoadImage), or drafted from a blank canvas + plate_prompt
  person_i     pose  -> stylize                      (one branch per rider, prompts from plan.json)
  compose_k    riders are added to the scene by Qwen, two per pass, back to front
               (the deployed pipeline pastes cutouts with exact code geometry instead; this preview
                is also the "group call" A/B variant)
  merch        last compose pass (no motion lines)
  hero         effects pass (speed lines)
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from souvenir.schemas import Analysis, Plan, ProductSpec, Slot

MODEL_FILES = {
    "unet": "qwen_image_edit_2511_fp8mixed.safetensors",
    "clip": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "vae": "qwen_image_vae.safetensors",
    "lora": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
}


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self._n = 0

    def add(self, cls: str, title: str | None = None, **inputs: Any) -> str:
        self._n += 1
        nid = str(self._n)
        self.nodes[nid] = {"class_type": cls, "inputs": inputs, "_meta": {"title": title or cls}}
        return nid

    @staticmethod
    def out(nid: str, idx: int = 0) -> list:
        return [nid, idx]


class QwenEditBuilder:
    """Adds one Qwen-Image-Edit-2511 edit (Lightning 4-step) per call, sharing loaders."""

    def __init__(self, g: Graph, steps: int = 4, cfg: float = 1.0, shift: float = 3.1, lightning: bool = True,
                 models: dict[str, str] | None = None):
        m = {**MODEL_FILES, **(models or {})}
        self.g = g
        unet = g.add("UNETLoader", "Load Qwen-Image-Edit-2511", unet_name=m["unet"], weight_dtype="default")
        self.clip = g.add("CLIPLoader", "Load text encoder", clip_name=m["clip"], type="qwen_image", device="default")
        self.vae = g.add("VAELoader", "Load VAE", vae_name=m["vae"])
        model = g.add("ModelSamplingAuraFlow", shift=shift, model=g.out(unet))
        model = g.add("CFGNorm", strength=1, pre_cfg=False, model=g.out(model))
        if lightning:
            model = g.add("LoraLoaderModelOnly", "Lightning 4-step LoRA", lora_name=m["lora"], strength_model=1, model=g.out(model))
        self.model = model
        self.steps = steps if lightning else max(steps, 40)
        self.cfg = cfg if lightning else 4.0

    def edit(self, images: list[list], prompt: str, seed: int, title: str, size: tuple[int, int] | None = None) -> list:
        g = self.g
        base = g.add("ImageScaleToTotalPixels", f"{title}: scale", upscale_method="lanczos", megapixels=1, resolution_steps=1, image=images[0])
        if size:
            base = g.add("ResizeAndPadImage", f"{title}: canvas", target_width=size[0], target_height=size[1],
                         padding_color="white", interpolation="lanczos", image=g.out(base))
        refs: dict[str, list] = {"image1": g.out(base)}
        for i, im in enumerate(images[1:3], start=2):
            refs[f"image{i}"] = im
        pos = g.add("TextEncodeQwenImageEditPlus", f"{title}: prompt", prompt=prompt, clip=g.out(self.clip), vae=g.out(self.vae), **refs)
        neg = g.add("TextEncodeQwenImageEditPlus", f"{title}: negative", prompt="", clip=g.out(self.clip), vae=g.out(self.vae), **refs)
        pos = g.add("FluxKontextMultiReferenceLatentMethod", reference_latents_method="index_timestep_zero", conditioning=g.out(pos))
        neg = g.add("FluxKontextMultiReferenceLatentMethod", reference_latents_method="index_timestep_zero", conditioning=g.out(neg))
        latent = g.add("VAEEncode", pixels=g.out(base), vae=g.out(self.vae))
        sample = g.add("KSampler", f"{title}: sample", seed=int(seed) % 2**31, steps=self.steps, cfg=self.cfg,
                       sampler_name="euler", scheduler="simple", denoise=1, model=g.out(self.model),
                       positive=g.out(pos), negative=g.out(neg), latent_image=g.out(latent))
        decoded = g.add("VAEDecode", f"{title}: decode", samples=g.out(sample), vae=g.out(self.vae))
        return g.out(decoded)


def where(slot: Slot) -> str:
    x = "on the left" if slot.cx < 0.38 else "on the right" if slot.cx > 0.62 else "in the center"
    y = "high up" if slot.cy < 0.42 else "low" if slot.cy > 0.6 else "at mid height"
    size = "large and close to the camera" if slot.width >= 0.4 else "small and far away" if slot.width <= 0.25 else "medium size"
    return f"{x}, {y}, {size}"


def _without_style_image(prompt: str, style_description: str) -> str:
    """Rewrite a stylize prompt for [posed, face] inputs when no style reference image exists."""
    style = style_description or "3D animated feature-film style"
    p = re.sub(r"in the (illustration )?style of image 2", f"in a {style}", prompt)
    return p.replace("image 3", "image 2")


def build_workflow(spec: ProductSpec, analysis: Analysis, plan: Plan, refs: dict[str, str],
                   lightning: bool = True, models: dict[str, str] | None = None, prefix: str = "souvenir") -> dict[str, Any]:
    """refs maps logical names -> what goes into LoadImage.image (filename in ComfyUI/input, or URL):
    plate (optional), style (optional), pose:<key>, body:<id>, face:<id>."""
    g = Graph()
    q = QwenEditBuilder(g, lightning=lightning, models=models)
    W, H = spec.canvas.width, spec.canvas.height
    load = lambda name, title: g.out(g.add("LoadImage", title, image=refs[name]))  # noqa: E731
    save = lambda img, name: g.add("SaveImage", f"save {name}", filename_prefix=f"{prefix}/{name}", images=img)  # noqa: E731

    style = load("style", "style reference") if "style" in refs else None

    # ---- plate
    if "plate" in refs:
        plate = load("plate", "background plate")
    else:
        blank = g.out(g.add("EmptyImage", "blank canvas", width=W, height=H, batch_size=1, color=0))
        if style is not None:
            prompt = spec.plate_prompt.replace("image 1", "image 2")
            plate = q.edit([blank, style], prompt, seed=plan.harmonize.seed if plan.harmonize else 1, title="plate", size=(W, H))
        else:
            prompt = spec.plate_prompt.replace("in the illustration style of image 1", f"in a {spec.style_description}")
            plate = q.edit([blank], prompt, seed=plan.harmonize.seed if plan.harmonize else 1, title="plate", size=(W, H))
    save(plate, "plate")

    # ---- riders
    riders: dict[int, list] = {}
    for pp in plan.per_person:
        pose_step, style_step = pp.steps[0], pp.steps[1]
        body = load(f"body:{pp.id}", f"person {pp.id} body crop")
        face = load(f"face:{pp.id}", f"person {pp.id} face crop")
        pose_ref = load(f"pose:{pp.pose_ref}", f"pose {pp.pose_ref}")
        posed = q.edit([body, pose_ref], pose_step.prompt, seed=pose_step.seed or 0, title=f"person {pp.id} pose")
        save(posed, f"person_{pp.id}_pose")
        if style is not None:
            styled = q.edit([posed, style, face], style_step.prompt, seed=style_step.seed or 0, title=f"person {pp.id} stylize")
        else:
            styled = q.edit([posed, face], _without_style_image(style_step.prompt, spec.style_description),
                            seed=style_step.seed or 0, title=f"person {pp.id} stylize")
        save(styled, f"person_{pp.id}_style")
        riders[pp.id] = styled

    # ---- compose, back to front, two riders per pass (Qwen takes 3 images: scene + 2 riders)
    order = sorted(plan.slots, key=lambda pid: plan.slots[pid].z)
    scene = plate
    seed0 = plan.harmonize.seed if plan.harmonize else 7
    for k in range(0, len(order), 2):
        group = order[k : k + 2]
        if len(group) == 2:
            a, b = group
            prompt = (f"Add the person from image 2 {where(plan.slots[a])}, and the person from image 3 "
                      f"{where(plan.slots[b])}, into the scene of image 1. ")
            imgs = [scene, riders[a], riders[b]]
        else:
            (a,) = group
            prompt = f"Add the person from image 2 into the scene of image 1, {where(plan.slots[a])}. "
            imgs = [scene, riders[a]]
        prompt += ("Each person hangs face down in a harness from their own zipline cable running from the upper "
                   "left to the lower right. Keep each person's face, pose, clothing and drawing style the same, and "
                   "keep everything already in image 1.")
        scene = q.edit(imgs, prompt, seed=seed0 + k, title=f"compose {k // 2 + 1}", size=(W, H))
        save(scene, f"compose_{k // 2 + 1}")
    save(scene, "merch")

    if plan.effects:
        hero = q.edit([scene], plan.effects.prompt, seed=plan.effects.seed or 0, title="effects", size=(W, H))
        save(hero, "hero")
    return g.nodes


def collect_inputs(spec: ProductSpec, analysis: Analysis, plan: Plan, out_dir: Path) -> dict[str, Path]:
    """Copy every image the workflow loads into out_dir/inputs with short, unique names."""
    inputs = out_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    tag = out_dir.name[:24]
    files: dict[str, Path] = {}

    def put(name: str, src: str | Path) -> None:
        src = Path(src)
        dst = inputs / f"{tag}_{name.replace(':', '_')}{src.suffix or '.png'}"
        shutil.copyfile(src, dst)
        files[name] = dst

    if spec.background_plates and Path(plan.plate).is_file():
        put("plate", plan.plate)
    if spec.style_refs and Path(spec.style_refs[0]).is_file():
        put("style", spec.style_refs[0])
    for key in {pp.pose_ref for pp in plan.per_person}:
        put(f"pose:{key}", spec.pose_refs[key])
    for pp in plan.per_person:
        put(f"body:{pp.id}", analysis.crops[str(pp.id)]["body"])
        put(f"face:{pp.id}", analysis.crops[str(pp.id)]["face"])
    return files


def check_graph(nodes: dict[str, dict[str, Any]]) -> list[str]:
    """Every link must point to an existing node. Returns a list of problems (empty = OK)."""
    problems = []
    for nid, node in nodes.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] not in nodes:
                problems.append(f"node {nid} ({node['class_type']}) input {name} -> missing node {value[0]}")
    return problems


def write(nodes: dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(nodes, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
