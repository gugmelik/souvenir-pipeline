"""Data models shared by every stage. Everything that is saved to a run folder is one of these."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- geometry


class Box(BaseModel):
    """Pixel box (x0, y0) top-left, (x1, y1) bottom-right."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def clip(self, width: int, height: int) -> "Box":
        return Box(
            x0=max(0.0, self.x0),
            y0=max(0.0, self.y0),
            x1=min(float(width), self.x1),
            y1=min(float(height), self.y1),
        )

    def scaled(self, factor: float) -> "Box":
        """Grow (or shrink) around the center."""
        hw, hh = self.w * factor / 2, self.h * factor / 2
        return Box(x0=self.cx - hw, y0=self.cy - hh, x1=self.cx + hw, y1=self.cy + hh)

    def square(self) -> "Box":
        side = max(self.w, self.h)
        return Box(x0=self.cx - side / 2, y0=self.cy - side / 2, x1=self.cx + side / 2, y1=self.cy + side / 2)

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return int(round(self.x0)), int(round(self.y0)), int(round(self.x1)), int(round(self.y1))


# --------------------------------------------------------------------------- product spec


class PromptTemplates(BaseModel):
    """Fallback prompt templates. The template planner fills them; the LLM planner may rewrite them.

    Placeholders: {clothing} {hair} {accessories} {arms} {expression} {age_group}
    """

    pose: str
    stylize: str
    harmonize: str
    effects: str
    pose_check_question: str = "Is the person lying horizontally, face down, in a flying pose?"


class Canvas(BaseModel):
    width: int = 1200
    height: int = 1600


class ProductSpec(BaseModel):
    """Everything that is fixed for one attraction product. Written once, reused for every guest."""

    product: str
    title: str = ""
    canvas: Canvas = Field(default_factory=Canvas)
    style_refs: list[str]
    style_description: str = ""
    background_plates: list[str]  # paths or globs, e.g. "assets/plates/approved/slotzilla_*.png"
    plate_prompt: str = ""  # used by `souvenir plates` to draft background candidates
    pose_refs: dict[str, str]
    formation_rules: dict[str, str]
    pose_by_formation: dict[str, list[str]] = Field(default_factory=dict)
    pose_arms: dict[str, str] = Field(default_factory=dict)  # pose ref key -> arms wording that matches the drawing
    expression_variants: list[str] = Field(default_factory=lambda: ["joyful laughing expression"])
    cable_direction: Literal["upper_left_to_lower_right", "upper_right_to_lower_left", "none"] = "upper_left_to_lower_right"
    cable_angle_deg: float = 22.0
    templates: PromptTemplates
    required_text: list[str] = Field(default_factory=list)
    banned_text_ocr: list[str] = Field(default_factory=list)
    banned_terms_in_prompts: list[str] = Field(default_factory=list)
    checklist: list[str] = Field(default_factory=list)
    min_people: int = 1
    max_people: int = 6
    notes: str = ""


# --------------------------------------------------------------------------- analysis


AgeGroup = Literal["child", "teen", "adult", "senior", "unknown"]


class DetectedPerson(BaseModel):
    id: int
    face_box: Box
    body_box: Box
    det_score: float = 1.0


class PersonCard(BaseModel):
    """What the VLM says a person looks like. Clothing/hair/accessories only: no race, ethnicity or skin."""

    id: int
    position_in_photo: str = ""
    relative_height: str = ""
    clothing: str = ""
    hair: str = ""
    accessories: list[str] = Field(default_factory=list)
    age_group: AgeGroup = "unknown"
    body_type: str = ""
    card_ok: bool = True  # False when the VLM failed and this is a placeholder


class Analysis(BaseModel):
    source: str
    width: int
    height: int
    n_people: int
    persons: list[DetectedPerson]
    cards: list[PersonCard]
    ignored_faces: int = 0  # small background faces dropped as bystanders
    crops: dict[str, dict[str, str]] = Field(default_factory=dict)  # person id -> {"body": path, "face": path}


# --------------------------------------------------------------------------- plan


Op = Literal["pose", "stylize", "cutout", "harmonize", "effects", "edit"]


class Step(BaseModel):
    op: Op
    # Symbolic inputs, resolved by the executor:
    #   "body_crop", "face_crop", "prev", "pose_ref", "style_ref", "composite", "merch", "source"
    inputs: list[str] = Field(default_factory=list)
    prompt: str = ""
    negative_prompt: str = " "
    seed: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class PersonPlan(BaseModel):
    id: int
    pose_ref: str
    arms: str = ""
    expression: str = ""
    steps: list[Step]


class Slot(BaseModel):
    """Where a rider goes on the canvas, normalized 0..1. width is rider width as a fraction of canvas width."""

    cx: float
    cy: float
    width: float
    z: int = 0  # draw order, low first (far away)


class Plan(BaseModel):
    product: str
    n_people: int
    formation: str
    plate: str
    planner: Literal["llm", "template"] = "template"
    per_person: list[PersonPlan]
    slots: dict[int, Slot]  # person id -> slot
    harmonize: Step | None
    effects: Step | None
    checks: list[str]
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- verification / reports


class CheckResult(BaseModel):
    name: str
    passed: bool | None  # None = could not be decided (e.g. detector failed on stylized art)
    score: float | None = None
    detail: str = ""


class VerifyReport(BaseModel):
    target: str
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(c.passed is not False for c in self.checks)


class StepRecord(BaseModel):
    person: int | None
    op: str
    attempt: int
    seed: int | None
    prompt: str
    output: str
    seconds: float
    check: CheckResult | None = None


class RunReport(BaseModel):
    run_id: str
    mode: str
    product: str
    source: str
    status: Literal["ok", "failed_checks", "unsupported", "error"]
    n_people: int = 0
    outputs: dict[str, str] = Field(default_factory=dict)
    steps: list[StepRecord] = Field(default_factory=list)
    verify: list[VerifyReport] = Field(default_factory=list)
    backend_calls: int = 0
    seconds_total: float = 0.0
    prompt_tokens: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str = ""
