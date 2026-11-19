"""Stage 4 - compositor: place N rider cutouts on the approved background plate, with one cable each.

This is plain image code, not AI: the number of riders, their order, size (depth) and the cable
geometry are exact. The harmonize pass afterwards only blends lighting.
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw

from souvenir.imaging import fit_cover, trim_alpha
from souvenir.schemas import Box, Canvas, Slot

SS = 2  # supersampling factor for anti-aliased cables


def _direction(cable_direction: str, angle_deg: float) -> tuple[float, float]:
    a = math.radians(angle_deg)
    if cable_direction == "upper_right_to_lower_left":
        return -math.cos(a), math.sin(a)
    return math.cos(a), math.sin(a)


def _draw_cable(size: tuple[int, int], anchor: tuple[float, float], harness_top: tuple[float, float],
                direction: tuple[float, float], width: float) -> Image.Image:
    W, H = size
    layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    L = 2 * max(W, H)
    ax, ay = anchor
    p0 = ((ax - direction[0] * L) * SS, (ay - direction[1] * L) * SS)
    p1 = ((ax + direction[0] * L) * SS, (ay + direction[1] * L) * SS)
    w = max(2, int(width * SS))
    d.line([p0, p1], fill=(35, 35, 45, 255), width=w)
    d.line([(p0[0], p0[1] - w * 0.25), (p1[0], p1[1] - w * 0.25)], fill=(190, 190, 210, 160), width=max(1, w // 3))
    # strap from cable down to the harness on the rider's back
    d.line([(ax * SS, ay * SS), (harness_top[0] * SS, harness_top[1] * SS)], fill=(30, 30, 30, 255), width=max(2, w // 2))
    trolley = w * 1.6
    d.ellipse([ax * SS - trolley, ay * SS - trolley, ax * SS + trolley, ay * SS + trolley], fill=(70, 70, 80, 255))
    return layer.resize((W, H), Image.LANCZOS)


def compose(
    plate: Image.Image,
    cutouts: dict[int, Image.Image],
    slots: dict[int, Slot],
    canvas: Canvas,
    cable_direction: str = "upper_left_to_lower_right",
    cable_angle_deg: float = 22.0,
) -> tuple[Image.Image, dict[int, Box]]:
    W, H = canvas.width, canvas.height
    out = fit_cover(plate.convert("RGB"), W, H).convert("RGBA")
    direction = _direction(cable_direction, cable_angle_deg)
    boxes: dict[int, Box] = {}

    for pid in sorted(slots, key=lambda k: slots[k].z):
        slot = slots[pid]
        rider = trim_alpha(cutouts[pid].convert("RGBA"))
        target_w = max(8, int(slot.width * W))
        scale = target_w / rider.width
        rider = rider.resize((target_w, max(8, int(rider.height * scale))), Image.LANCZOS)
        x0 = int(slot.cx * W - rider.width / 2)
        y0 = int(slot.cy * H - rider.height / 2)
        boxes[pid] = Box(x0=x0, y0=y0, x1=x0 + rider.width, y1=y0 + rider.height).clip(W, H)

        if cable_direction != "none":
            harness_top = (x0 + rider.width / 2, y0 + rider.height * 0.15)
            strap = max(12, rider.height * 0.22)
            anchor = (harness_top[0], harness_top[1] - strap)
            cable = _draw_cable((W, H), anchor, harness_top, direction, width=3 + 6 * slot.width)
            out.alpha_composite(cable)

        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        layer.paste(rider, (x0, y0), rider)
        out.alpha_composite(layer)

    return out.convert("RGB"), boxes
