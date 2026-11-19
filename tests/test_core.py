"""Fast tests that need no GPU, network or API keys."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from souvenir.compositor import compose
from souvenir.config import PROJECT_ROOT, load_product
from souvenir.formation import assign_slots, layout, select_formation
from souvenir.identity import match
from souvenir.planner import LLMPlanner, prompt_problems, template_plan
from souvenir.schemas import Analysis, Box, Canvas, DetectedPerson, PersonCard
from souvenir.tokens import budget_report
from souvenir.vlm import extract_json

SPEC_PATH = PROJECT_ROOT / "configs" / "products" / "slotzilla_flight_v1.json"
RULES = {"1": "solo_hero", "2": "race_mode", "3": "staggered_line", "4+": "v_formation"}


def _analysis(n: int) -> Analysis:
    persons, cards = [], []
    for i in range(1, n + 1):
        face = Box(x0=100 * i, y0=100, x1=100 * i + 60, y1=160)
        persons.append(DetectedPerson(id=i, face_box=face, body_box=face.scaled(4)))
        cards.append(PersonCard(id=i, clothing=f"red shirt {i}", hair="short dark hair", accessories=["sunglasses"], age_group="adult"))
    crops = {str(i): {"body": "b.png", "face": "f.png"} for i in range(1, n + 1)}
    return Analysis(source="x", width=1200, height=900, n_people=n, persons=persons, cards=cards, crops=crops)


@pytest.mark.parametrize("n,expected", [(1, "solo_hero"), (2, "race_mode"), (3, "staggered_line"), (4, "v_formation"), (6, "v_formation")])
def test_select_formation(n, expected):
    assert select_formation(n, RULES) == expected


@pytest.mark.parametrize("formation", ["solo_hero", "race_mode", "staggered_line", "v_formation"])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_layout_has_one_slot_per_person_inside_canvas(formation, n):
    slots = layout(formation, n)
    assert len(slots) == n
    for s in slots:
        assert 0 < s.cx < 1 and 0 < s.cy < 1 and 0 < s.width < 1


def test_assign_slots_keeps_left_to_right_order():
    slots = layout("staggered_line", 3)
    assigned = assign_slots([1, 2, 3], slots)
    assert assigned[1].cx < assigned[2].cx < assigned[3].cx


def test_template_plan_prompts_are_short_and_clean():
    spec = load_product(SPEC_PATH)
    plan = template_plan(spec, _analysis(3), seed=1)
    assert plan.n_people == 3 and plan.formation == "staggered_line"
    assert [s.op for s in plan.per_person[0].steps] == ["pose", "stylize", "cutout"]
    for pp in plan.per_person:
        for step in pp.steps[:2]:
            assert "red shirt" in step.prompt  # clothing quoted from the card
            assert prompt_problems(step.prompt, spec, 180) == [], step.prompt
    for step in (plan.harmonize, plan.effects):
        assert prompt_problems(step.prompt, spec, 180) == [], step.prompt


def test_prompt_problems_catches_brands_negations_and_length():
    spec = load_product(SPEC_PATH)
    assert any("banned" in p for p in prompt_problems("Draw it in Disney style", spec, 120))
    assert any("negative" in p for p in prompt_problems("Do not add the Golden Nugget logo", spec, 120))
    assert any("too long" in p for p in prompt_problems("word " * 400, spec, 120))


class FakeVLM:
    name = "fake"

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def chat_json(self, system, user, images=None):
        self.calls += 1
        return self.reply


def test_llm_planner_applies_good_prompts_and_falls_back_on_bad_ones():
    spec = load_product(SPEC_PATH)
    good = "Show the person from image 1 flying horizontally in a zipline harness, pose of image 2, red shirt 1."
    reply = {
        "persons": [
            {"id": 1, "pose_prompt": good, "stylize_prompt": "Redraw image 1 in the style of image 2, face of image 3."},
            {"id": 2, "pose_prompt": "Make it Disney style", "stylize_prompt": "Redraw image 1 in the style of image 2."},
        ],
        "harmonize_prompt": "Blend the riders into the neon scene.",
        "effects_prompt": "Add speed lines.",
    }
    vlm = FakeVLM(reply)
    plan = LLMPlanner(vlm, max_prompt_tokens=120, attempts=2).plan(spec, _analysis(2), seed=0)
    assert vlm.calls == 2  # retried once with feedback
    assert plan.per_person[0].steps[0].prompt == good
    assert "Disney" not in plan.per_person[1].steps[0].prompt  # fell back to the template
    assert any("person 2 pose_prompt" in w for w in plan.warnings)


def test_compose_places_every_rider():
    plate = Image.new("RGB", (800, 600), (40, 20, 80))
    cut = Image.new("RGBA", (200, 120), (0, 0, 0, 0))
    cut.paste(Image.new("RGBA", (160, 80), (255, 0, 0, 255)), (20, 20))
    slots = assign_slots([1, 2, 3], layout("staggered_line", 3))
    out, boxes = compose(plate, {1: cut, 2: cut, 3: cut}, slots, Canvas(width=600, height=800))
    assert out.size == (600, 800)
    assert sorted(boxes) == [1, 2, 3]
    front = max(slots, key=lambda k: slots[k].width)
    assert boxes[front].w == max(b.w for b in boxes.values())


def test_hungarian_match():
    scores = np.array([[0.1, 0.9], [0.8, 0.2]])
    pairs = sorted(match(scores))
    assert [(r, c) for r, c, _ in pairs] == [(0, 1), (1, 0)]


def test_brief_is_over_qwen_default_limit():
    text = (PROJECT_ROOT / "briefs" / "slotzilla_flight_brief.md").read_text(encoding="utf-8")
    report = budget_report(text)
    assert report["tokens"] > 512 and report["over_diffusers_default"]


def test_extract_json_tolerates_fences():
    assert extract_json('Sure!\n```json\n{"a": {"b": 1}}\n```') == {"a": {"b": 1}}


def test_demo_runs_end_to_end(tmp_path: Path):
    from souvenir.demo import run_demo

    assert run_demo(tmp_path, people=3) == 0
    reports = list((tmp_path / "runs").glob("*/report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert report["n_people"] == 3
    for key in ("composite", "merch", "hero"):
        assert Path(report["outputs"][key]).is_file()
    # 3 people x (pose + stylize) + harmonize + effects
    assert report["backend_calls"] == 8


def test_eval_compares_pipeline_and_single_prompt(tmp_path: Path):
    from souvenir.cli import MOCK_ALL
    from souvenir.config import Config
    from souvenir.demo import placeholder_plate, placeholder_style, synthetic_group
    from souvenir.evaluate import run_eval
    from souvenir.pipeline import Pipeline, build_components
    from souvenir.poses import write_all

    photo, boxes = synthetic_group(2)
    photo.save(tmp_path / "p.png")
    placeholder_plate().save(tmp_path / "plate.png")
    placeholder_style().save(tmp_path / "style.png")
    poses = write_all(tmp_path / "poses")

    cfg = Config.load(None, MOCK_ALL)
    spec = load_product(SPEC_PATH)
    spec.background_plates = [str(tmp_path / "plate.png")]
    spec.style_refs = [str(tmp_path / "style.png")]
    spec.pose_refs = {p.stem: str(p) for p in poses}
    pipe = Pipeline(cfg, spec, build_components(cfg, spec, mock_boxes=boxes))
    brief = (PROJECT_ROOT / "briefs" / "slotzilla_flight_brief.md").read_text(encoding="utf-8")

    csv_path = run_eval(pipe, [str(tmp_path / "p.png")], ["pipeline", "single:main"], tmp_path / "eval", brief)
    rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3  # header + 2 modes
    assert (tmp_path / "eval" / "summary.md").is_file() and (tmp_path / "eval" / "index.html").is_file()
    single = json.loads(next((tmp_path / "eval" / "runs").glob("*single*/report.json")).read_text(encoding="utf-8"))
    assert single["prompt_tokens"]["brief"] > 512
    assert any("512-token" in w for w in single["warnings"])
