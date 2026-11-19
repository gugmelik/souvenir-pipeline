"""Stage 2 - planner: product spec + person cards -> a short, step-by-step plan.

Code decides the structure (which steps, which input images, formation, slots, seeds).
The LLM only writes the prompt text for each step, and every prompt is validated:
  * <= max_prompt_tokens (so nothing is silently truncated by the text encoder)
  * no brand / IP names from spec.banned_terms_in_prompts
  * positive phrasing (negations like "do not add Golden Nugget" tend to ADD the concept)
If the LLM output fails validation twice, that prompt falls back to the spec template.
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any

from souvenir.config import read_prompt
from souvenir.formation import assign_slots, layout, select_formation
from souvenir.schemas import Analysis, PersonCard, PersonPlan, Plan, ProductSpec, Step
from souvenir.tokens import count_tokens
from souvenir.vlm import VLM

log = logging.getLogger(__name__)

_NEGATION = re.compile(r"\b(do not|don't|never|avoid|must not|no extra)\b", re.I)


# --------------------------------------------------------------------------- validation


def prompt_problems(
    prompt: str, spec: ProductSpec, max_tokens: int, tokenizer: str | None = None
) -> list[str]:
    problems = []
    n = count_tokens(prompt, tokenizer)
    if n > max_tokens:
        problems.append(f"too long: {n} tokens > {max_tokens}")
    low = prompt.lower()
    for term in spec.banned_terms_in_prompts:
        if term.lower() in low:
            problems.append(f"contains banned term '{term}'")
    if _NEGATION.search(prompt):
        problems.append("uses negative phrasing; describe what should be there instead")
    if not prompt.strip():
        problems.append("empty prompt")
    return problems


# --------------------------------------------------------------------------- template filling


def _accessories_clause(card: PersonCard) -> str:
    if not card.accessories:
        return ""
    return "Keep the " + ", ".join(card.accessories) + "."


def fill(template: str, card: PersonCard | None = None, arms: str = "", expression: str = "") -> str:
    values = {
        "clothing": (card.clothing if card and card.clothing else "the same outfit"),
        "hair": (card.hair if card and card.hair else "the same hairstyle"),
        "accessories": _accessories_clause(card) if card else "",
        "age_group": card.age_group if card else "",
        "arms": arms,
        "expression": expression,
    }
    out = template
    for k, v in values.items():
        out = out.replace("{" + k + "}", v)
    return re.sub(r"\s+", " ", out).strip()


# --------------------------------------------------------------------------- structure (code)


def _skeleton(spec: ProductSpec, analysis: Analysis, seed: int) -> Plan:
    rng = random.Random(seed)
    n = analysis.n_people
    formation = select_formation(n, spec.formation_rules)
    slots = assign_slots([p.id for p in analysis.persons], layout(formation, n))
    pose_keys = spec.pose_by_formation.get(formation) or list(spec.pose_refs.keys())
    plate = spec.background_plates[rng.randrange(len(spec.background_plates))] if spec.background_plates else ""

    per_person = []
    for i, card in enumerate(analysis.cards):
        pose_key = pose_keys[i % len(pose_keys)]
        arms = spec.pose_arms.get(pose_key, "")
        expression = spec.expression_variants[i % len(spec.expression_variants)]
        base = seed + 1000 * card.id
        per_person.append(
            PersonPlan(
                id=card.id,
                pose_ref=pose_key,
                arms=arms,
                expression=expression,
                steps=[
                    Step(op="pose", inputs=["body_crop", "pose_ref"], prompt=fill(spec.templates.pose, card, arms, expression), seed=base + 1),
                    Step(op="stylize", inputs=["prev", "style_ref", "face_crop"], prompt=fill(spec.templates.stylize, card, arms, expression), seed=base + 2),
                    Step(op="cutout", inputs=["prev"]),
                ],
            )
        )

    warnings = []
    if analysis.ignored_faces:
        warnings.append(f"{analysis.ignored_faces} small background face(s) ignored as bystanders")
    warnings += [f"person {c.id}: VLM person card failed, using generic wording" for c in analysis.cards if not c.card_ok]

    return Plan(
        product=spec.product,
        n_people=n,
        formation=formation,
        plate=plate,
        planner="template",
        per_person=per_person,
        slots=slots,
        harmonize=Step(op="harmonize", inputs=["composite"], prompt=fill(spec.templates.harmonize), seed=seed + 7),
        effects=Step(op="effects", inputs=["merch"], prompt=fill(spec.templates.effects), seed=seed + 8),
        checks=["rider_count", "identity_per_rider", "vlm_checklist", "ocr_text"],
        warnings=warnings,
    )


# --------------------------------------------------------------------------- planners


def template_plan(spec: ProductSpec, analysis: Analysis, seed: int = 0) -> Plan:
    return _skeleton(spec, analysis, seed)


class LLMPlanner:
    def __init__(self, vlm: VLM, max_prompt_tokens: int = 120, tokenizer: str | None = None, attempts: int = 2):
        self.vlm = vlm
        self.max_prompt_tokens = max_prompt_tokens
        self.tokenizer = tokenizer
        self.attempts = attempts

    def _request(self, spec: ProductSpec, plan: Plan, analysis: Analysis) -> dict[str, Any]:
        cards = {c.id: c for c in analysis.cards}
        persons = []
        for pp in plan.per_person:
            persons.append(
                {
                    "id": pp.id,
                    "card": cards[pp.id].model_dump(exclude={"card_ok"}),
                    "arms": pp.arms,
                    "expression": pp.expression,
                    "pose_step_images": {"image 1": "photo crop of this person", "image 2": "pose reference drawing"},
                    "stylize_step_images": {"image 1": "result of the pose step", "image 2": "house style reference", "image 3": "original face crop"},
                    "template_pose": pp.steps[0].prompt,
                    "template_stylize": pp.steps[1].prompt,
                }
            )
        return {
            "product": spec.title or spec.product,
            "style_description": spec.style_description,
            "formation": plan.formation,
            "n_people": plan.n_people,
            "max_tokens_per_prompt": self.max_prompt_tokens,
            "banned_terms": spec.banned_terms_in_prompts,
            "persons": persons,
            "template_harmonize": plan.harmonize.prompt if plan.harmonize else "",
            "template_effects": plan.effects.prompt if plan.effects else "",
        }

    def plan(self, spec: ProductSpec, analysis: Analysis, seed: int = 0) -> Plan:
        plan = _skeleton(spec, analysis, seed)
        system = read_prompt("planner_system.md")
        request = self._request(spec, plan, analysis)
        feedback = ""
        reply: dict[str, Any] | None = None
        problems: dict[str, list[str]] = {}

        for attempt in range(self.attempts):
            user = json.dumps(request, ensure_ascii=False, indent=2)
            if feedback:
                user += "\n\nYour previous answer had these problems, fix them:\n" + feedback
            try:
                reply = self.vlm.chat_json(system, user)
            except Exception as exc:
                plan.warnings.append(f"LLM planner unavailable ({exc}); using templates")
                return plan
            problems = self._check(reply, spec, plan)
            if not problems:
                break
            feedback = "\n".join(f"- {k}: {'; '.join(v)}" for k, v in problems.items())
            log.info("planner attempt %d problems: %s", attempt + 1, feedback)

        self._apply(reply or {}, plan, problems)
        plan.planner = "llm"
        return plan

    def _check(self, reply: dict[str, Any], spec: ProductSpec, plan: Plan) -> dict[str, list[str]]:
        problems: dict[str, list[str]] = {}
        by_id = {int(p.get("id", -1)): p for p in reply.get("persons", []) if isinstance(p, dict)}
        for pp in plan.per_person:
            entry = by_id.get(pp.id)
            if not entry:
                problems[f"person {pp.id}"] = ["missing from 'persons'"]
                continue
            for key in ("pose_prompt", "stylize_prompt"):
                p = prompt_problems(str(entry.get(key, "")), spec, self.max_prompt_tokens, self.tokenizer)
                if p:
                    problems[f"person {pp.id} {key}"] = p
        for key in ("harmonize_prompt", "effects_prompt"):
            p = prompt_problems(str(reply.get(key, "")), spec, self.max_prompt_tokens, self.tokenizer)
            if p:
                problems[key] = p
        return problems

    def _apply(self, reply: dict[str, Any], plan: Plan, problems: dict[str, list[str]]) -> None:
        by_id = {int(p.get("id", -1)): p for p in reply.get("persons", []) if isinstance(p, dict)}
        for pp in plan.per_person:
            entry = by_id.get(pp.id, {})
            for step, key in ((pp.steps[0], "pose_prompt"), (pp.steps[1], "stylize_prompt")):
                label = f"person {pp.id} {key}"
                if label in problems or f"person {pp.id}" in problems:
                    plan.warnings.append(f"{label}: kept template ({'; '.join(problems.get(label, ['missing']))})")
                else:
                    step.prompt = str(entry[key]).strip()
        for step, key in ((plan.harmonize, "harmonize_prompt"), (plan.effects, "effects_prompt")):
            if step is None:
                continue
            if key in problems:
                plan.warnings.append(f"{key}: kept template ({'; '.join(problems[key])})")
            else:
                step.prompt = str(reply[key]).strip()


def make_plan(
    spec: ProductSpec, analysis: Analysis, vlm: VLM, mode: str = "llm", seed: int = 0,
    max_prompt_tokens: int = 120, tokenizer: str | None = None,
) -> Plan:
    if mode == "template":
        return template_plan(spec, analysis, seed)
    return LLMPlanner(vlm, max_prompt_tokens=max_prompt_tokens, tokenizer=tokenizer).plan(spec, analysis, seed)
