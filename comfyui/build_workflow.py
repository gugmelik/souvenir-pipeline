"""Build the occo souvenir ComfyUI workflow in both formats.

  workflows/occo_souvenir_slotzilla.json      editor format: open with ComfyUI -> Workflow -> Open, or drag in
  workflows/occo_souvenir_slotzilla_api.json  API format: POST /prompt, comfyui-api run.py --workflow-json

Usage (ComfyUI running with the occo nodes installed, to read node definitions):
  python comfyui/build_workflow.py --comfy http://127.0.0.1:8188
or offline with the cached node definitions:
  python comfyui/build_workflow.py
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "object_info_cache.json"
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}

MODELS = {
    "unet": "qwen_image_edit_2511_fp8mixed.safetensors",
    "clip": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "vae": "qwen_image_vae.safetensors",
    "lora": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
}
VLM = {"vlm_base_url": "", "vlm_model": "Qwen/Qwen3-VL-8B-Instruct", "vlm_api_key": "EMPTY"}


# --------------------------------------------------------------------------- API graph


class G:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.group: dict[str, str] = {}

    def add(self, nid: str, cls: str, title: str, group: str, **inputs):
        self.nodes[nid] = {"class_type": cls, "inputs": inputs, "_meta": {"title": title}}
        self.group[nid] = group
        return [nid, 0]


def qwen_edit(g: G, p: str, title: str, group: str, image1, prompt, seed: int, extra_images=()):
    """Core Qwen-Image-Edit-2511 edit: same nodes as comfyui-api/workflows/qwen_image_edit_2511.json."""
    refs = {"image1": image1}
    for i, im in enumerate(extra_images, start=2):
        refs[f"image{i}"] = im
    g.add(f"{p}1", "TextEncodeQwenImageEditPlus", f"{title}: prompt", group, clip=["2", 0], prompt=prompt, vae=["3", 0], **refs)
    g.add(f"{p}2", "TextEncodeQwenImageEditPlus", f"{title}: negative", group, clip=["2", 0], prompt="", vae=["3", 0], **refs)
    g.add(f"{p}3", "FluxKontextMultiReferenceLatentMethod", f"{title}: ref method", group, conditioning=[f"{p}1", 0], reference_latents_method="index_timestep_zero")
    g.add(f"{p}4", "FluxKontextMultiReferenceLatentMethod", f"{title}: ref method (neg)", group, conditioning=[f"{p}2", 0], reference_latents_method="index_timestep_zero")
    g.add(f"{p}5", "VAEEncode", f"{title}: encode", group, pixels=image1, vae=["3", 0])
    g.add(f"{p}6", "KSampler", f"{title}: sample", group, seed=seed, steps=4, cfg=1.0, sampler_name="euler", scheduler="simple",
          denoise=1.0, model=["6", 0], positive=[f"{p}3", 0], negative=[f"{p}4", 0], latent_image=[f"{p}5", 0])
    return g.add(f"{p}7", "VAEDecode", f"{title}: decode", group, samples=[f"{p}6", 0], vae=["3", 0])


def build_api() -> G:
    g = G()
    M = "Models (Qwen-Image-Edit-2511 + Lightning 4-step)"
    g.add("1", "UNETLoader", "Load Qwen-Image-Edit-2511", M, unet_name=MODELS["unet"], weight_dtype="default")
    g.add("2", "CLIPLoader", "Load text encoder", M, clip_name=MODELS["clip"], type="qwen_image", device="default")
    g.add("3", "VAELoader", "Load VAE", M, vae_name=MODELS["vae"])
    g.add("4", "ModelSamplingAuraFlow", "AuraFlow shift", M, model=["1", 0], shift=3.1)
    g.add("5", "CFGNorm", "CFGNorm", M, model=["4", 0], strength=1.0, pre_cfg=False)
    g.add("6", "LoraLoaderModelOnly", "Lightning 4-step LoRA", M, model=["5", 0], lora_name=MODELS["lora"], strength_model=1.0)

    I = "1. Guest photo -> people -> plan"
    g.add("10", "LoadImage", "Guest photo", I, image="guest_photo.jpg")
    g.add("11", "OccoProductSpec", "Product spec (Slotzilla)", I, spec_path="configs/products/slotzilla_flight_v1.json")
    g.add("12", "OccoDetectPeople", "Detect people", I, image=["10", 0], detector="insightface", min_face_ratio=0.35,
          manual_faces="", rekognition_region="us-west-2", **VLM)
    g.add("13", "PreviewImage", "Rider crops", I, images=["12", 2])
    g.add("14", "PreviewAny", "Person cards", I, source=["12", 3])
    g.add("15", "OccoPlan", "Plan", I, spec=["11", 0], people=["12", 0], seed=7, planner="template", max_prompt_tokens=120, **VLM)
    g.add("16", "PreviewAny", "Plan (all prompts)", I, source=["15", 1])

    P = "2. Background plate (draft; replace with an approved plate via LoadImage)"
    g.add("20", "EmptyImage", "Blank canvas 1200x1600", P, width=1200, height=1600, batch_size=1, color=0x1E1430)
    plate = qwen_edit(g, "2", "plate", P, ["20", 0], ["11", 1], seed=1001)
    g.add("28", "SaveImage", "Save plate", P, images=plate, filename_prefix="souvenir/plate")

    R = "3. Riders: pose -> stylize (loop) -> cutout -> compose"
    g.add("30", "OccoQwenEditRiders", "Qwen edit riders", R, model=["6", 0], clip=["2", 0], vae=["3", 0], spec=["11", 0],
          people=["12", 0], plan=["15", 0], steps=4, cfg=1.0, sampler_name="euler", scheduler="simple",
          reference_method="index_timestep_zero")
    g.add("31", "PreviewImage", "Posed riders", R, images=["30", 1])
    g.add("32", "PreviewImage", "Stylized riders", R, images=["30", 2])
    g.add("33", "OccoCutout", "Cutout", R, riders=["30", 0], method="auto", white_tolerance=38)
    g.add("34", "PreviewImage", "Cutouts", R, images=["33", 1])
    g.add("35", "OccoCompose", "Compose", R, spec=["11", 0], plan=["15", 0], riders=["33", 0], plate=plate)
    g.add("36", "SaveImage", "Save composite", R, images=["35", 0], filename_prefix="souvenir/composite")

    H = "4. Harmonize (merch) and effects (hero)"
    merch = qwen_edit(g, "4", "harmonize", H, ["35", 0], ["15", 2], seed=1002)
    g.add("48", "SaveImage", "Save merch", H, images=merch, filename_prefix="souvenir/merch")
    hero = qwen_edit(g, "5", "effects", H, merch, ["15", 3], seed=1003)
    g.add("58", "SaveImage", "Save hero", H, images=hero, filename_prefix="souvenir/hero")

    V = "5. Checks"
    g.add("60", "OccoVerify", "Verify hero", V, image=hero, spec=["11", 0], people=["12", 0], layout=["35", 1],
          identity="insightface", identity_threshold=0.30, **VLM)
    g.add("61", "PreviewAny", "Verify report", V, source=["60", 0])
    return g


# --------------------------------------------------------------------------- editor (UI) format


def _spec_list(info: dict) -> list[tuple[str, list, bool]]:
    """(name, spec, optional) in the order the frontend creates them."""
    order = info.get("input_order") or {k: list(v) for k, v in info["input"].items() if k != "hidden"}
    out = []
    for section in ("required", "optional"):
        for name in order.get(section, []):
            out.append((name, info["input"][section][name], section == "optional"))
    return out


def _is_widget(spec: list) -> bool:
    t = spec[0]
    if isinstance(t, list):
        return True
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    return t in WIDGET_TYPES and not opts.get("forceInput")


def _depths(api: dict) -> dict[str, int]:
    memo: dict[str, int] = {}

    def depth(nid: str) -> int:
        if nid not in memo:
            memo[nid] = 0
            ups = [v[0] for v in api[nid]["inputs"].values() if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)]
            memo[nid] = 1 + max((depth(u) for u in ups), default=-1)
        return memo[nid]

    for nid in api:
        depth(nid)
    return memo


def to_ui(g: G, object_info: dict) -> dict:
    api = g.nodes
    ids = {nid: i + 1 for i, nid in enumerate(api)}
    nodes, links = [], []
    link_id = 0
    pending = []
    for nid, node in api.items():
        info = object_info[node["class_type"]]
        inputs, widgets = [], []
        for name, spec, optional in _spec_list(info):
            value = node["inputs"].get(name)
            linked = isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if _is_widget(spec):
                if linked:
                    link_id += 1
                    inputs.append({"name": name, "type": spec[0] if isinstance(spec[0], str) else "COMBO",
                                   "widget": {"name": name}, "link": link_id})
                    pending.append((link_id, value, nid, len(inputs) - 1, spec[0] if isinstance(spec[0], str) else "COMBO"))
                    widgets.append(opts.get("default", ""))
                else:
                    widgets.append(value if value is not None else opts.get("default"))
                if opts.get("control_after_generate"):
                    widgets.append("fixed")
                if opts.get("image_upload"):
                    widgets.append("image")
            else:
                entry = {"name": name, "type": spec[0], "link": None}
                if linked:
                    link_id += 1
                    entry["link"] = link_id
                    pending.append((link_id, value, nid, len(inputs), spec[0]))
                elif optional:
                    entry["shape"] = 7
                inputs.append(entry)
        outputs = [{"name": (info.get("output_name") or info["output"])[i], "type": t, "links": [], "slot_index": i}
                   for i, t in enumerate(info["output"])]
        nodes.append({"id": ids[nid], "type": node["class_type"], "title": node["_meta"]["title"], "pos": [0, 0],
                      "size": [340, 120], "flags": {}, "order": 0, "mode": 0, "inputs": inputs, "outputs": outputs,
                      "properties": {"Node name for S&R": node["class_type"]}, "widgets_values": widgets,
                      "_api_id": nid})
    by_api = {n["_api_id"]: n for n in nodes}
    for lid, (src, slot), dst, dst_slot, typ in pending:
        links.append([lid, ids[src], slot, ids[dst], dst_slot, typ])
        by_api[src]["outputs"][slot]["links"].append(lid)

    # layout: one horizontal band (LiteGraph group) per stage; inside a band, columns follow graph depth
    depth = _depths(api)
    group_order = list(dict.fromkeys(g.group[nid] for nid in api))
    by_group: dict[str, list[dict]] = defaultdict(list)
    for n in nodes:
        by_group[g.group[n["_api_id"]]].append(n)
    groups, y0 = [], 40
    big = ("PreviewImage", "SaveImage", "PreviewAny")
    for gname in group_order:
        members = by_group[gname]
        cols = sorted({depth[n["_api_id"]] for n in members})
        col_of = {d: i for i, d in enumerate(cols)}
        y_col: dict[int, int] = defaultdict(int)
        for n in sorted(members, key=lambda n: (depth[n["_api_id"]], n["id"])):
            c = col_of[depth[n["_api_id"]]]
            h = 70 + 24 * len(n["inputs"]) + 28 * len(n["widgets_values"])
            if n["type"] in big:
                h = max(h, 420)
            if n["type"] in ("OccoDetectPeople", "OccoPlan", "OccoVerify"):
                h += 80
            n["pos"] = [60 + c * 460, y0 + 60 + y_col[c]]
            n["size"] = [420, h]
            y_col[c] += h + 50
        height = max(y_col.values()) + 70
        width = 60 + len(cols) * 460
        groups.append({"title": gname, "bounding": [30, y0, width, height], "color": "#3f789e", "font_size": 24, "flags": {}})
        y0 += height + 60
    order = sorted(nodes, key=lambda n: depth[n["_api_id"]])
    for i, n in enumerate(order):
        n["order"] = i
    for n in nodes:
        n.pop("_api_id")

    return {"last_node_id": len(nodes), "last_link_id": link_id, "nodes": nodes, "links": links, "groups": groups,
            "config": {}, "extra": {"ds": {"scale": 0.6, "offset": [0, 0]}}, "version": 0.4}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default=None, help="running ComfyUI URL to read /object_info from (refreshes the cache)")
    ap.add_argument("--out", default=str(HERE / "workflows"))
    a = ap.parse_args()
    g = build_api()
    classes = sorted({n["class_type"] for n in g.nodes.values()})
    if a.comfy:
        info = {}
        for c in classes:
            with urllib.request.urlopen(f"{a.comfy.rstrip('/')}/object_info/{c}", timeout=30) as r:
                info.update(json.loads(r.read()))
        missing = [c for c in classes if c not in info]
        if missing:
            raise SystemExit(f"ComfyUI does not know: {missing} (are the occo nodes installed?)")
        CACHE.write_text(json.dumps(info, indent=1), encoding="utf-8")
    info = json.loads(CACHE.read_text(encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "occo_souvenir_slotzilla_api.json").write_text(json.dumps(g.nodes, indent=2), encoding="utf-8")
    (out / "occo_souvenir_slotzilla.json").write_text(json.dumps(to_ui(g, info), indent=1), encoding="utf-8")
    print(f"wrote {out / 'occo_souvenir_slotzilla.json'} and {out / 'occo_souvenir_slotzilla_api.json'} ({len(g.nodes)} nodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
