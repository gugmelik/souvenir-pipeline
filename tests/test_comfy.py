"""ComfyUI export: graph is well-formed and has the expected stages."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from souvenir.cli import MOCK_ALL
from souvenir.comfy import build_workflow, check_graph, collect_inputs
from souvenir.config import PROJECT_ROOT, Config, load_product
from souvenir.demo import placeholder_plate, placeholder_style, synthetic_group
from souvenir.pipeline import Pipeline, build_components
from souvenir.planner import template_plan
from souvenir.analyzer import analyze
from souvenir.poses import write_all


def _setup(tmp_path: Path, n: int, with_assets: bool):
    photo, boxes = synthetic_group(n)
    spec = load_product(PROJECT_ROOT / "configs" / "products" / "slotzilla_flight_v1.json")
    poses = write_all(tmp_path / "poses")
    spec.pose_refs = {p.stem: str(p) for p in poses}
    if with_assets:
        placeholder_plate().save(tmp_path / "plate.png")
        placeholder_style().save(tmp_path / "style.png")
        spec.background_plates, spec.style_refs = [str(tmp_path / "plate.png")], [str(tmp_path / "style.png")]
    else:
        spec.background_plates, spec.style_refs = [], []
    cfg = Config.load(None, MOCK_ALL)
    pipe = Pipeline(cfg, spec, build_components(cfg, spec, mock_boxes=boxes))
    out = tmp_path / "run"
    analysis = analyze(photo, pipe.c.detector, pipe.c.vlm, out)
    plan = template_plan(pipe.spec, analysis, seed=3)
    return pipe.spec, analysis, plan, out


@pytest.mark.parametrize("n", [1, 3, 4])
@pytest.mark.parametrize("with_assets", [True, False])
def test_workflow_is_well_formed(tmp_path, n, with_assets):
    spec, analysis, plan, out = _setup(tmp_path, n, with_assets)
    files = collect_inputs(spec, analysis, plan, out)
    assert ("plate" in files) == with_assets and ("style" in files) == with_assets
    refs = {k: v.name for k, v in files.items()}
    nodes = build_workflow(spec, analysis, plan, refs)
    assert check_graph(nodes) == []
    edits = sum(1 for x in nodes.values() if x["class_type"] == "KSampler")
    plate_edit = 0 if with_assets else 1
    assert edits == plate_edit + 2 * n + math.ceil(n / 2) + 1
    saves = {x["inputs"]["filename_prefix"].split("/")[-1] for x in nodes.values() if x["class_type"] == "SaveImage"}
    assert {"plate", "merch", "hero", "person_1_pose", "person_1_style"} <= saves
    loads = [x["inputs"]["image"] for x in nodes.values() if x["class_type"] == "LoadImage"]
    assert all(name in refs.values() for name in loads)
    for x in nodes.values():  # model files match comfyui-api's workflow
        if x["class_type"] == "UNETLoader":
            assert x["inputs"]["unet_name"] == "qwen_image_edit_2511_fp8mixed.safetensors"
        if x["class_type"] == "TextEncodeQwenImageEditPlus":
            assert "image4" not in x["inputs"]


def test_full_comfyui_workflow_builds_and_converts():
    import importlib.util
    import json

    path = PROJECT_ROOT / "comfyui" / "build_workflow.py"
    spec_ = importlib.util.spec_from_file_location("build_workflow", path)
    bw = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(bw)
    g = bw.build_api()
    assert check_graph(g.nodes) == []
    info = json.loads((PROJECT_ROOT / "comfyui" / "object_info_cache.json").read_text(encoding="utf-8"))
    assert {n["class_type"] for n in g.nodes.values()} <= set(info)
    ui = bw.to_ui(g, info)
    assert len(ui["nodes"]) == len(g.nodes) and ui["groups"]
    link_ids = {link[0] for link in ui["links"]}
    for n in ui["nodes"]:
        for inp in n["inputs"]:
            assert inp["link"] is None or inp["link"] in link_ids
