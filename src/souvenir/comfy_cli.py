"""`souvenir comfy`: analyze + plan one photo, then write a ComfyUI workflow to preview the result."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from souvenir.analyzer import analyze
from souvenir.comfy import build_workflow, check_graph, collect_inputs, write
from souvenir.config import Config
from souvenir.imaging import load_image, save
from souvenir.planner import make_plan
from souvenir.schemas import ProductSpec


def _uploader(comfyui_api: Path):
    """Use comfyui-api's own uploader (same bucket/prefix and IAM policy as its workflows)."""
    scripts = comfyui_api / "scripts"
    if not (scripts / "upload_input.py").is_file():
        raise SystemExit(f"{scripts / 'upload_input.py'} not found; pass --comfyui-api <dir> or use --refs local")
    sys.path.insert(0, str(scripts))
    from upload_input import upload  # type: ignore

    def up(path: Path) -> str:
        info = upload(path)
        return info.get("public_url") or info.get("url")

    return up


def export_comfy(args: Any, cfg: Config, spec: ProductSpec, pipe: Any) -> int:
    img = load_image(args.image)
    stem = Path(str(args.image).split("?")[0]).stem[:32] or "photo"
    out = Path(args.out) / "comfy" / f"{datetime.now():%Y%m%d_%H%M%S}_{stem}"
    out.mkdir(parents=True, exist_ok=True)
    save(img, out / "source.png")

    analysis = analyze(img, pipe.c.detector, pipe.c.vlm, out, source=str(args.image))
    if analysis.n_people == 0:
        print("no people detected; nothing to export")
        return 1
    if analysis.n_people > spec.max_people:
        print(f"{analysis.n_people} people detected; product supports up to {spec.max_people}")
        return 1
    plan = make_plan(spec, analysis, pipe.c.vlm, mode=cfg.get("planner.mode", "llm"), seed=args.seed or 0,
                     max_prompt_tokens=int(cfg.get("planner.max_prompt_tokens", 120)), tokenizer=cfg.get("planner.tokenizer"))
    (out / "analysis.json").write_text(json.dumps(analysis.model_dump(), indent=2, default=str), encoding="utf-8")
    (out / "plan.json").write_text(json.dumps(plan.model_dump(), indent=2, default=str), encoding="utf-8")

    files = collect_inputs(spec, analysis, plan, out)
    refs_mode = "upload" if args.run else args.refs
    if refs_mode == "upload":
        up = _uploader(Path(args.comfyui_api))
        refs = {name: up(path) for name, path in files.items()}
    else:
        refs = {name: path.name for name, path in files.items()}

    nodes = build_workflow(spec, analysis, plan, refs, lightning=not args.no_lightning, prefix=f"souvenir/{out.name}")
    problems = check_graph(nodes)
    if problems:
        print("workflow has broken links:\n  " + "\n  ".join(problems))
        return 1
    wf = write(nodes, out / "workflow_api.json")
    edits = sum(1 for n in nodes.values() if n["class_type"] == "KSampler")

    print(f"\npeople: {analysis.n_people}   formation: {plan.formation}   planner: {plan.planner}   "
          f"Qwen edits: {edits}   nodes: {len(nodes)}")
    for w in plan.warnings:
        print(f"warning: {w}")
    if "plate" not in files:
        print("note: no approved plate yet -> the workflow drafts one from a blank canvas (stage 'plate')")
    if "style" not in files:
        print("note: no house-style reference yet -> stylize uses the text style description only")
    print(f"\nworkflow: {wf}")
    if refs_mode == "local":
        print(f"inputs:   {out / 'inputs'}  -> copy these files into ComfyUI\\input, then drag workflow_api.json into ComfyUI")
    run_cmd = [sys.executable, "scripts/run.py", "qwen_image_edit_2511", "--workflow-json", str(wf.resolve())]
    print("vast:     cd " + str(Path(args.comfyui_api)) + " ; python scripts\\run.py qwen_image_edit_2511 --workflow-json " + str(wf.resolve()))

    if args.run:
        venv_py = Path(args.comfyui_api) / ".venv" / "Scripts" / "python.exe"
        if venv_py.is_file():
            run_cmd[0] = str(venv_py)
        print("\nsending to Vast via comfyui-api ...")
        return subprocess.call(run_cmd, cwd=args.comfyui_api)
    return 0
