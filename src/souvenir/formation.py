"""Formation choice and rider layout. Deterministic code, so 'Auto-Select or Flag' never reaches a prompt.

Coordinates are normalized to the canvas (0..1). Larger `width` = closer to the camera.
Cables run from the upper-left (launch tower) to the lower-right by default.
"""

from __future__ import annotations

from souvenir.schemas import Slot


def select_formation(n: int, rules: dict[str, str]) -> str:
    """rules like {"1": "solo_hero", "2": "race_mode", "3": "staggered_line", "4+": "v_formation"}."""
    if str(n) in rules:
        return rules[str(n)]
    best_key, best_min = None, -1
    for key, value in rules.items():
        if key.endswith("+"):
            lo = int(key[:-1])
            if n >= lo and lo > best_min:
                best_key, best_min = value, lo
    if best_key:
        return best_key
    return "staggered_line"


def _staggered(n: int) -> list[Slot]:
    # Leader lower-right and biggest; followers climb back up toward the tower, smaller.
    if n == 1:
        return [Slot(cx=0.52, cy=0.52, width=0.62, z=1)]
    start, end = (0.70, 0.70), (0.30, 0.28)
    w_front, w_back = 0.46 if n <= 3 else 0.38, 0.24 if n <= 3 else 0.18
    slots = []
    for i in range(n):
        t = i / (n - 1)
        slots.append(
            Slot(
                cx=start[0] + (end[0] - start[0]) * t,
                cy=start[1] + (end[1] - start[1]) * t,
                width=w_front + (w_back - w_front) * t,
                z=n - i,  # front rider drawn last
            )
        )
    return slots


def _race(n: int) -> list[Slot]:
    if n != 2:
        return _staggered(n)
    return [Slot(cx=0.33, cy=0.50, width=0.42, z=1), Slot(cx=0.68, cy=0.60, width=0.44, z=2)]


def _v_formation(n: int) -> list[Slot]:
    if n < 3:
        return _staggered(n)
    w0 = 0.36 if n <= 5 else 0.30
    slots = [Slot(cx=0.50, cy=0.66, width=w0, z=100)]
    rows = (n - 1 + 1) // 2
    step_x, step_y = min(0.17, 0.38 / rows), min(0.13, 0.40 / rows)
    row = 1
    remaining = n - 1
    while remaining > 0:
        w = w0 * (0.82**row)
        dx, dy = step_x * row, step_y * row
        y = 0.66 - dy
        if remaining >= 2:
            slots.append(Slot(cx=0.50 - dx, cy=y, width=w, z=100 - row))
            slots.append(Slot(cx=0.50 + dx, cy=y, width=w, z=100 - row))
            remaining -= 2
        else:
            slots.append(Slot(cx=0.50 - dx, cy=y, width=w, z=100 - row))
            remaining -= 1
        row += 1
    return slots


def _solo(n: int) -> list[Slot]:
    return _staggered(1) if n == 1 else _staggered(n)


LAYOUTS = {
    "solo_hero": _solo,
    "staggered_line": _staggered,
    "race_mode": _race,
    "v_formation": _v_formation,
}


def layout(formation: str, n: int) -> list[Slot]:
    fn = LAYOUTS.get(formation, _staggered)
    slots = fn(n)
    assert len(slots) == n, f"layout {formation} produced {len(slots)} slots for {n} people"
    return slots


def assign_slots(person_ids_left_to_right: list[int], slots: list[Slot]) -> dict[int, Slot]:
    """Keep the family's left-to-right order: leftmost person gets the leftmost slot."""
    ordered = sorted(slots, key=lambda s: s.cx)
    return {pid: slot for pid, slot in zip(person_ids_left_to_right, ordered)}
