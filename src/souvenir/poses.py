"""Draw simple pose-reference images (OpenPose-like colored skeletons) for the zipline 'superhero' poses.

Image models follow a pose drawing far better than a paragraph about "prone, horizontal, facing
downward and forward". Regenerate or replace these with your own photos of a real rider.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

LIMB_W = 26
COLORS = {
    "torso": (255, 85, 0),
    "arm_r": (255, 170, 0),
    "arm_l": (170, 255, 0),
    "leg_r": (0, 255, 170),
    "leg_l": (0, 170, 255),
    "head": (255, 0, 85),
}

BASE = {
    "head": (760, 330),
    "neck": (700, 352),
    "hip": (430, 382),
    "knee_r": (300, 358), "ankle_r": (165, 388),
    "knee_l": (305, 400), "ankle_l": (172, 426),
}

ARMS = {
    "prone_superman": {"elbow_r": (805, 338), "wrist_r": (905, 318), "elbow_l": (795, 380), "wrist_l": (898, 372)},
    "prone_celebrate": {"elbow_r": (705, 262), "wrist_r": (745, 175), "elbow_l": (660, 268), "wrist_l": (632, 182)},
    "prone_point": {"elbow_r": (805, 338), "wrist_r": (910, 322), "elbow_l": (720, 440), "wrist_l": (790, 488)},
}

LABELS = {
    "prone_superman": "prone flying pose, body horizontal, arms stretched forward",
    "prone_celebrate": "prone flying pose, body horizontal, arms raised in a cheer",
    "prone_point": "prone flying pose, one arm forward, one arm pointing to the side",
}


def draw_pose(name: str, size: tuple[int, int] = (1024, 768)) -> Image.Image:
    pts = {**BASE, **ARMS[name]}
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    # zipline cable + harness strap, so the model sees where the body hangs
    d.line([(0, 120), (size[0], 260)], fill=(120, 120, 130), width=8)
    strap_x = 560
    cable_y = 120 + 140 * strap_x / size[0]
    d.line([(strap_x, cable_y), (strap_x, 360)], fill=(60, 60, 60), width=10)

    def limb(a: str, b: str, color: tuple[int, int, int]) -> None:
        d.line([pts[a], pts[b]], fill=color, width=LIMB_W)
        for p in (pts[a], pts[b]):
            d.ellipse([p[0] - 14, p[1] - 14, p[0] + 14, p[1] + 14], fill=color)

    limb("hip", "knee_l", COLORS["leg_l"])
    limb("knee_l", "ankle_l", COLORS["leg_l"])
    limb("neck", "elbow_l", COLORS["arm_l"])
    limb("elbow_l", "wrist_l", COLORS["arm_l"])
    limb("neck", "hip", COLORS["torso"])
    limb("hip", "knee_r", COLORS["leg_r"])
    limb("knee_r", "ankle_r", COLORS["leg_r"])
    limb("neck", "elbow_r", COLORS["arm_r"])
    limb("elbow_r", "wrist_r", COLORS["arm_r"])
    hx, hy = pts["head"]
    d.ellipse([hx - 46, hy - 46, hx + 46, hy + 46], fill=COLORS["head"])
    d.text((20, size[1] - 30), f"pose reference: {LABELS[name]}", fill=(80, 80, 80))
    return img


def write_all(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ARMS:
        p = out_dir / f"{name}.png"
        draw_pose(name).save(p)
        paths.append(p)
    return paths
