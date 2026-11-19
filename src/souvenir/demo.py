"""Offline demo: synthetic group photo + placeholder assets + mock models -> a full run folder.

Shows the plumbing (analysis.json, plan.json, steps, composite, merch, hero, report.json)
without a GPU or API keys. Images are obviously fake; the point is the structure.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from souvenir.cli import MOCK_ALL
from souvenir.config import PROJECT_ROOT, Config, load_product
from souvenir.poses import write_all

PALETTE = [(220, 60, 60), (60, 120, 220), (60, 170, 90), (230, 160, 40), (150, 80, 200), (40, 170, 170)]


def synthetic_group(n: int, size: tuple[int, int] = (1200, 900)) -> tuple[Image.Image, list[tuple[float, float, float, float]]]:
    W, H = size
    img = Image.new("RGB", size, (235, 232, 225))
    d = ImageDraw.Draw(img)
    boxes = []
    gap = W / (n + 1)
    for i in range(n):
        cx = gap * (i + 1)
        head_r = 48 - 4 * (i % 2)
        top = 220 + 30 * (i % 2)
        d.rectangle([cx - 85, top + head_r * 2 + 10, cx + 85, H - 60], fill=PALETTE[i % len(PALETTE)])
        d.ellipse([cx - head_r, top, cx + head_r, top + 2 * head_r], fill=(240, 200, 170))
        d.ellipse([cx - 18, top + 30, cx - 8, top + 40], fill=(30, 30, 30))
        d.ellipse([cx + 8, top + 30, cx + 18, top + 40], fill=(30, 30, 30))
        boxes.append((cx - head_r, top, cx + head_r, top + 2 * head_r))
    return img, boxes


def placeholder_plate(size: tuple[int, int] = (1200, 1600)) -> Image.Image:
    W, H = size
    img = Image.new("RGB", size)
    d = ImageDraw.Draw(img)
    for y in range(H):
        t = y / H
        d.line([(0, y), (W, y)], fill=(int(40 + 90 * t), int(10 + 20 * t), int(80 + 60 * t)))
    d.rectangle([60, 100, 220, 900], fill=(90, 70, 110))  # launch tower
    d.rectangle([0, int(H * 0.82), W, H], fill=(200, 60, 160))  # LED canopy
    d.text((W * 0.55, H * 0.12), "LAS VEGAS 2026", fill=(255, 230, 120))
    return img


def placeholder_style(size: tuple[int, int] = (512, 512)) -> Image.Image:
    img = Image.new("RGB", size, (250, 245, 235))
    d = ImageDraw.Draw(img)
    d.ellipse([140, 80, 372, 312], fill=(255, 205, 160), outline=(60, 40, 30), width=6)
    d.text((150, 420), "house style reference (placeholder)", fill=(60, 40, 30))
    return img


def run_demo(out: Path, people: int = 3) -> int:
    from souvenir.pipeline import Pipeline, build_components

    out = out if out.is_absolute() else PROJECT_ROOT / out
    assets = out / "assets"
    (assets / "plates").mkdir(parents=True, exist_ok=True)
    photo, boxes = synthetic_group(people)
    photo_path = out / "synthetic_group.png"
    photo.save(photo_path)
    placeholder_plate().save(assets / "plates" / "plate_demo.png")
    placeholder_style().save(assets / "style.png")
    pose_paths = write_all(assets / "poses")

    cfg = Config.load(None, MOCK_ALL)
    spec = load_product(PROJECT_ROOT / "configs" / "products" / "slotzilla_flight_v1.json")
    spec.background_plates = [str(assets / "plates" / "plate_demo.png")]
    spec.style_refs = [str(assets / "style.png")]
    spec.pose_refs = {p.stem: str(p) for p in pose_paths}

    pipe = Pipeline(cfg, spec, build_components(cfg, spec, mock_boxes=boxes))
    report = pipe.run(str(photo_path), out / "runs", seed=42)
    print(f"status={report.status} people={report.n_people} calls={report.backend_calls}")
    for k, v in report.outputs.items():
        print(f"  {k:<10} {v}")
    for w in report.warnings:
        print(f"  warning: {w}")
    if report.error:
        print(f"  error: {report.error}")
    return 0 if report.status in ("ok", "failed_checks") else 1
