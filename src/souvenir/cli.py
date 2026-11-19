"""Command line entry point:  python -m souvenir <command> ...

  demo      offline end-to-end run with mock models (no GPU, no API keys)
  tokens    count tokens in a brief and show where Qwen's limit cuts it
  poses     draw pose-reference images into assets/poses
  plan      analyze a photo and print the plan (no image generation)
  run       full pipeline (or a single-prompt baseline) on one photo
  eval      batch comparison of modes over a folder of photos
  plates    draft background-plate candidates for human approval
  comfy     export the plan for one photo as a ComfyUI workflow (preview before deploying)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from souvenir.config import PROJECT_ROOT, Config, load_product

DEFAULT_PRODUCT = "configs/products/slotzilla_flight_v1.json"

MOCK_ALL = {
    "backend.kind": "mock", "vlm.kind": "mock", "detector.kind": "mock", "segmenter.kind": "mock",
    "identity.kind": "mock", "ocr.enabled": False, "planner.mode": "template", "baselines": {},
}


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="pipeline.yaml (default configs/pipeline.yaml)")
    p.add_argument("--product", default=DEFAULT_PRODUCT)
    p.add_argument("--out", default="runs")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--mock-backend", action="store_true", help="mock only the image model (real detector + VLM)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set backend.kind=modal_comfy")


def _overrides(args: argparse.Namespace) -> dict:
    o: dict = {}
    if getattr(args, "mock_backend", False):
        o["backend.kind"] = "mock"
    for item in getattr(args, "set", []):
        k, v = item.split("=", 1)
        o[k] = json.loads(v) if v[:1] in "[{0123456789" or v in ("true", "false", "null") else v
    return o


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="souvenir", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="offline mock run")
    d.add_argument("--out", default="runs/demo")
    d.add_argument("--people", type=int, default=3)

    t = sub.add_parser("tokens", help="token count of a brief")
    t.add_argument("--file", required=True)
    t.add_argument("--tokenizer", default=None, help="e.g. Qwen/Qwen2.5-VL-7B-Instruct for an exact count")

    po = sub.add_parser("poses", help="draw pose references")
    po.add_argument("--out", default="assets/poses")

    pl = sub.add_parser("plan", help="analyze + plan only")
    pl.add_argument("--image", required=True)
    _common(pl)

    r = sub.add_parser("run", help="run one photo")
    r.add_argument("--image", required=True, help="path or URL")
    r.add_argument("--mode", default="pipeline", help="pipeline | single:main | single:gemini")
    r.add_argument("--brief", default=None, help="text file with the long prompt (for single:* modes)")
    _common(r)

    e = sub.add_parser("eval", help="batch comparison")
    e.add_argument("--images", required=True, help="folder, a .txt list of paths/URLs, or one image")
    e.add_argument("--modes", default="pipeline,single:main")
    e.add_argument("--brief", default="briefs/slotzilla_flight_brief.md")
    e.add_argument("--limit", type=int, default=0)
    _common(e)

    cf = sub.add_parser("comfy", help="export a ComfyUI workflow for one photo")
    cf.add_argument("--image", required=True, help="path or URL")
    cf.add_argument("--refs", choices=["local", "upload"], default="local",
                    help="local: LoadImage uses file names (copy inputs\\* to ComfyUI\\input); "
                         "upload: upload inputs with comfyui-api and put URLs in LoadImage")
    cf.add_argument("--comfyui-api", default="C:/dev/occo/comfyui-api", help="your comfyui-api project")
    cf.add_argument("--run", action="store_true", help="send it to your Vast endpoint via comfyui-api run.py (implies --refs upload)")
    cf.add_argument("--no-lightning", action="store_true", help="40-step sampling without the Lightning LoRA")
    _common(cf)

    pt = sub.add_parser("plates", help="draft background plates")
    pt.add_argument("--n", type=int, default=6)
    _common(pt)

    args = ap.parse_args(argv)

    if args.cmd == "demo":
        from souvenir.demo import run_demo

        return run_demo(Path(args.out), args.people)

    if args.cmd == "tokens":
        from souvenir.tokens import budget_report

        print(json.dumps(budget_report(Path(args.file).read_text(encoding="utf-8"), args.tokenizer), indent=2))
        return 0

    if args.cmd == "poses":
        from souvenir.poses import write_all

        out = Path(args.out)
        out = out if out.is_absolute() else PROJECT_ROOT / out
        for p in write_all(out):
            print(p)
        return 0

    cfg = Config.load(args.config, _overrides(args))
    spec = load_product(args.product)

    if args.cmd == "plates":
        from souvenir.backends import build_backend
        from souvenir.ocr import build_ocr
        from souvenir.pipeline import expand
        from souvenir.plates import generate_plates

        spec.style_refs = expand(spec.style_refs)
        kind = cfg.get("backend.kind")
        backend = build_backend(kind, cfg.section(f"backend.{kind}"))
        out = PROJECT_ROOT / "assets" / "plates" / "candidates"
        for res in generate_plates(spec, backend, build_ocr(cfg.section("ocr")), out, n=args.n, seed=args.seed or 0):
            print(json.dumps(res))
        print(f"\nReview {out}, copy the good ones to assets/plates/approved/")
        return 0

    from souvenir.pipeline import Pipeline, build_components

    pipe = Pipeline(cfg, spec, build_components(cfg, spec))

    if args.cmd == "comfy":
        from souvenir.comfy_cli import export_comfy

        return export_comfy(args, cfg, spec, pipe)

    if args.cmd == "plan":
        from souvenir.analyzer import analyze
        from souvenir.imaging import load_image
        from souvenir.planner import make_plan

        out = Path(args.out) / "plan_only"
        out.mkdir(parents=True, exist_ok=True)
        analysis = analyze(load_image(args.image), pipe.c.detector, pipe.c.vlm, out, source=args.image)
        if analysis.n_people == 0:
            print("no people detected")
            return 1
        plan = make_plan(spec, analysis, pipe.c.vlm, mode=cfg.get("planner.mode", "llm"), seed=args.seed or 0,
                         max_prompt_tokens=int(cfg.get("planner.max_prompt_tokens", 120)),
                         tokenizer=cfg.get("planner.tokenizer"))
        print(json.dumps({"analysis": analysis.model_dump(), "plan": plan.model_dump()}, indent=2, default=str))
        return 0

    if args.cmd == "run":
        brief = Path(args.brief).read_text(encoding="utf-8") if args.brief else None
        report = pipe.run(args.image, Path(args.out), mode=args.mode, seed=args.seed, brief=brief)
        print(json.dumps({"status": report.status, "n_people": report.n_people, "outputs": report.outputs,
                          "calls": report.backend_calls, "seconds": report.seconds_total,
                          "warnings": report.warnings, "error": report.error}, indent=2))
        return 0 if report.status in ("ok", "failed_checks") else 1

    if args.cmd == "eval":
        from souvenir.evaluate import list_images, run_eval

        images = list_images(args.images)
        if args.limit:
            images = images[: args.limit]
        brief_path = Path(args.brief)
        brief_path = brief_path if brief_path.is_absolute() else PROJECT_ROOT / brief_path
        brief = brief_path.read_text(encoding="utf-8") if brief_path.is_file() else None
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        csv_path = run_eval(pipe, images, modes, Path(args.out) / "eval", brief, seed=args.seed or 0)
        print(f"\nresults: {csv_path}\nsummary: {csv_path.parent / 'summary.md'}\nreview:  {csv_path.parent / 'index.html'}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
