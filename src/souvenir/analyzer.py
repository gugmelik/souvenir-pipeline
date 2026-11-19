"""Stage 1 - analyzer: count people in code, crop each one, and ask the VLM for a 'person card'.

The person card describes clothing, hair and accessories so prompts can quote them word for word.
It never describes race, ethnicity or skin: identity is carried by the face crop image, not by words.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from souvenir.config import read_prompt
from souvenir.detect import Detector
from souvenir.imaging import crop, save
from souvenir.schemas import Analysis, PersonCard
from souvenir.vlm import VLM

log = logging.getLogger(__name__)

_FORBIDDEN_KEYS = {"race", "ethnicity", "skin", "skin_tone", "skin_color", "nationality", "religion"}
_AGE_GROUPS = {"child", "teen", "adult", "senior"}


def _position_label(index: int, n: int) -> str:
    if n == 1:
        return "only person"
    if index == 0:
        return "far left"
    if index == n - 1:
        return "far right"
    return "center" if n == 3 else f"position {index + 1} from left"


def _height_labels(face_tops: list[float]) -> list[str]:
    if len(face_tops) == 1:
        return [""]
    order = sorted(range(len(face_tops)), key=lambda i: face_tops[i])  # smaller y = taller (roughly)
    labels = ["medium height"] * len(face_tops)
    labels[order[0]] = "tallest"
    labels[order[-1]] = "shortest"
    return labels


def _clean_card(raw: dict[str, Any], pid: int, position: str, height: str) -> PersonCard:
    raw = {k: v for k, v in raw.items() if k.lower() not in _FORBIDDEN_KEYS}
    acc = raw.get("accessories") or []
    if isinstance(acc, str):
        acc = [a.strip() for a in acc.split(",") if a.strip()]
    age = str(raw.get("age_group", "unknown")).lower().strip()
    return PersonCard(
        id=pid,
        position_in_photo=position,
        relative_height=height,
        clothing=str(raw.get("clothing", "")).strip(),
        hair=str(raw.get("hair", "")).strip(),
        accessories=[str(a) for a in acc][:6],
        age_group=age if age in _AGE_GROUPS else "unknown",  # type: ignore[arg-type]
        body_type=str(raw.get("body_type", "")).strip(),
        card_ok=True,
    )


def analyze(img: Image.Image, detector: Detector, vlm: VLM, out_dir: Path, source: str = "") -> Analysis:
    people, ignored = detector.people(img)
    n = len(people)
    crops_dir = out_dir / "crops"
    crops: dict[str, dict[str, str]] = {}
    cards: list[PersonCard] = []
    heights = _height_labels([p.face_box.y0 for p in people]) if people else []
    system = read_prompt("analyzer_person_card.md")

    for idx, person in enumerate(people):
        body = crop(img, person.body_box.scaled(1.06))
        face = crop(img, person.face_box.scaled(1.8).square())
        crops[str(person.id)] = {
            "body": save(body, crops_dir / f"person_{person.id}_body.png"),
            "face": save(face, crops_dir / f"person_{person.id}_face.png"),
        }
        position = _position_label(idx, n)
        user = (
            f"This crop shows person {person.id} of {n} in a group photo ({position}).\n"
            "Describe only this person and return the JSON object."
        )
        try:
            raw = vlm.chat_json(system, user, images=[body])
            card = _clean_card(raw, person.id, position, heights[idx])
        except Exception as exc:  # keep going with a placeholder; the plan notes it
            log.warning("person card failed for person %s: %s", person.id, exc)
            card = PersonCard(id=person.id, position_in_photo=position, relative_height=heights[idx], card_ok=False)
        cards.append(card)

    return Analysis(
        source=source,
        width=img.width,
        height=img.height,
        n_people=n,
        persons=people,
        cards=cards,
        ignored_faces=ignored,
        crops=crops,
    )
