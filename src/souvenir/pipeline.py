"""Orchestrator: analyzer -> planner -> per-person executor -> compositor -> harmonize/effects -> verifier.

Every run writes a self-contained folder:
  analysis.json, plan.json, report.json, crops/, steps/, composite.png, merch.png, hero.png
"""

from __future__ import annotations

import glob
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from souvenir.analyzer import analyze
from souvenir.backends import CountingBackend, build_backend
from souvenir.compositor import compose
from souvenir.config import Config, resolve_asset
from souvenir.detect import Detector, build_detector
from souvenir.identity import IdentityScorer, build_scorer
from souvenir.imaging import load_image, save
from souvenir.ocr import build_ocr
from souvenir.planner import make_plan
from souvenir.schemas import (
    Analysis, CheckResult, ProductSpec, RunReport, Step, StepRecord,
)
from souvenir.segment import Segmenter, build_segmenter
from souvenir.tokens import budget_report
from souvenir.verifier import Verifier
from souvenir.vlm import VLM, build_vlm

log = logging.getLogger(__name__)


@dataclass
class Components:
    detector: Detector
    vlm: VLM
    backend: CountingBackend
    segmenter: Segmenter
    scorer: IdentityScorer
    verifier: Verifier
    baseline_backends: dict[str, CountingBackend]


class _Unavailable:
    max_images = 4

    def __init__(self, kind: str, exc: Exception):
        self.name = f"{kind} (unavailable)"
        self.exc = exc

    def edit(self, images, prompt, **kw):
        raise RuntimeError(f"image backend could not be created: {self.exc}")


def expand(paths: list[str]) -> list[str]:
    """Resolve asset paths/globs relative to the project root."""
    out: list[str] = []
    for p in paths:
        full = str(resolve_asset(p))
        matches = sorted(glob.glob(full)) if any(ch in p for ch in "*?[") else [full]
        out.extend(matches)
    return out


def build_components(cfg: Config, spec: ProductSpec, mock_boxes: list[tuple[float, float, float, float]] | None = None) -> Components:
    vlm = build_vlm(cfg.section("vlm"))
    detector = build_detector(cfg.section("detector"), mock_boxes=mock_boxes)
    kind = cfg.get("backend.kind", "mock")
    try:
        backend = CountingBackend(build_backend(kind, cfg.section(f"backend.{kind}")))
    except Exception as exc:  # e.g. no pod running yet: `plan` still works, generation raises later
        backend = CountingBackend(_Unavailable(kind, exc))
    segmenter = build_segmenter(cfg.section("segmenter"))
    id_section = cfg.section("identity")
    scorer = build_scorer(id_section, vlm)
    thresholds = id_section.get("thresholds", {}) or {}
    threshold = float(thresholds.get(scorer.name, 0.5))
    verifier = Verifier(spec, vlm, scorer, detector, build_ocr(cfg.section("ocr")), identity_threshold=threshold,
                        checklist_min_yes=float(cfg.get("verify.checklist_min_yes", 1.0)))
    baselines: dict[str, CountingBackend] = {}
    for name, section in (cfg.section("baselines") or {}).items():
        if section and section.get("enabled"):
            try:
                baselines[name] = CountingBackend(build_backend(section.get("kind", name), section))
            except Exception as exc:  # a missing API key should not stop the main pipeline
                log.warning("baseline %s disabled: %s", name, exc)
    return Components(detector, vlm, backend, segmenter, scorer, verifier, baselines)


class Pipeline:
    def __init__(self, cfg: Config, spec: ProductSpec, comps: Components):
        self.cfg = cfg
        self.spec = spec
        self.c = comps
        self.retries = cfg.section("retries")
        self.spec.background_plates = expand(spec.background_plates)
        self.spec.style_refs = expand(spec.style_refs)
        self.spec.pose_refs = {k: str(resolve_asset(v)) for k, v in spec.pose_refs.items()}

    # ------------------------------------------------------------------ public

    def run(self, source: str, out_root: Path, mode: str = "pipeline", seed: int | None = None,
            brief: str | None = None) -> RunReport:
        seed = seed if seed is not None else random.randrange(2**31)
        img = load_image(source)
        stem = Path(str(source).split("?")[0]).stem[:40] or "photo"
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in f"{stem}_{mode}")  # Windows-safe
        run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{safe}"
        out = Path(out_root) / run_id
        out.mkdir(parents=True, exist_ok=True)
        save(img, out / "source.png")
        t0 = time.perf_counter()
        report = RunReport(run_id=run_id, mode=mode, product=self.spec.product, source=str(source), status="ok")
        calls_before = self._calls()
        try:
            analysis = analyze(img, self.c.detector, self.c.vlm, out, source=str(source))
            _dump(out / "analysis.json", analysis.model_dump())
            report.n_people = analysis.n_people
            if not (self.spec.min_people <= analysis.n_people <= self.spec.max_people):
                report.status = "unsupported"
                report.warnings.append(
                    f"{analysis.n_people} people detected; product supports {self.spec.min_people}-{self.spec.max_people}. "
                    "Route to a fallback product or human review."
                )
            elif mode == "pipeline":
                self._run_pipeline(img, analysis, out, seed, report)
            else:
                self._run_single_prompt(mode, img, analysis, out, seed, brief, report)
        except Exception as exc:
            log.exception("run failed")
            report.status = "error"
            report.error = f"{type(exc).__name__}: {exc}"
        report.seconds_total = round(time.perf_counter() - t0, 2)
        report.backend_calls = self._calls() - calls_before
        _dump(out / "report.json", report.model_dump())
        return report

    # ------------------------------------------------------------------ pipeline mode

    def _run_pipeline(self, img: Image.Image, analysis: Analysis, out: Path, seed: int, report: RunReport) -> None:
        plan = make_plan(
            self.spec, analysis, self.c.vlm,
            mode=self.cfg.get("planner.mode", "llm"), seed=seed,
            max_prompt_tokens=int(self.cfg.get("planner.max_prompt_tokens", 120)),
            tokenizer=self.cfg.get("planner.tokenizer"),
        )
        _dump(out / "plan.json", plan.model_dump())
        report.warnings += plan.warnings
        if not self.spec.background_plates:
            raise RuntimeError("no background plate found; add approved plates (see assets/README.md)")
        if not self.spec.style_refs:
            raise RuntimeError("no style reference found; add house-style images (see assets/README.md)")

        style_ref = load_image(self.spec.style_refs[0])
        cutouts: dict[int, Image.Image] = {}
        steps_dir = out / "steps"

        for pp in plan.per_person:
            crops = analysis.crops[str(pp.id)]
            body, face = load_image(crops["body"]), load_image(crops["face"])
            pose_ref = load_image(self.spec.pose_refs[pp.pose_ref])
            prev: Image.Image = body
            for step in pp.steps:
                named = {"body_crop": body, "face_crop": face, "prev": prev, "pose_ref": pose_ref, "style_ref": style_ref}
                if step.op == "cutout":
                    t = time.perf_counter()
                    rgba = self.c.segmenter.cutout(prev)
                    path = save(rgba, steps_dir / f"person_{pp.id}_cutout.png")
                    report.steps.append(StepRecord(person=pp.id, op="cutout", attempt=0, seed=None, prompt="",
                                                   output=path, seconds=round(time.perf_counter() - t, 2)))
                    cutouts[pp.id] = rgba
                    continue
                images = [named[k] for k in step.inputs]
                check = self._person_check(step.op, pp.id, face)
                prev = self._edit_with_retries(step, images, check, int(self.retries.get(step.op, 1)),
                                               steps_dir / f"person_{pp.id}_{step.op}", pp.id, report)

        composite, boxes = compose(
            load_image(plan.plate), cutouts, plan.slots, self.spec.canvas,
            self.spec.cable_direction, self.spec.cable_angle_deg,
        )
        report.outputs["composite"] = save(composite, out / "composite.png")
        _dump(out / "layout.json", {str(k): v.model_dump() for k, v in boxes.items()})

        W, H = self.spec.canvas.width, self.spec.canvas.height
        n = analysis.n_people
        count_check = lambda im: self.c.verifier.check_count(im, n)  # noqa: E731

        merch = composite
        if plan.harmonize and self.cfg.get("harmonize.enabled", True):
            merch = self._edit_with_retries(plan.harmonize, [composite], count_check, int(self.retries.get("harmonize", 1)),
                                            steps_dir / "harmonize", None, report, size=(W, H), fallback=composite)
        report.outputs["merch"] = save(merch, out / "merch.png")

        hero = merch
        if plan.effects and self.cfg.get("effects.enabled", True):
            hero = self._edit_with_retries(plan.effects, [merch], count_check, int(self.retries.get("effects", 1)),
                                           steps_dir / "effects", None, report, size=(W, H), fallback=merch)
        report.outputs["hero"] = save(hero, out / "hero.png")

        for name, image in (("merch", merch), ("hero", hero)):
            report.verify.append(self.c.verifier.verify_composite(name, image, analysis, boxes))
        report.prompt_tokens = {
            f"person_{pp.id}_{s.op}": int(budget_report(s.prompt, self.cfg.get("planner.tokenizer"))["tokens"])
            for pp in plan.per_person for s in pp.steps if s.prompt
        }
        if self.c.backend.dropped_images:
            report.warnings.append(f"backend {self.c.backend.name} accepts {self.c.backend.max_images} images; "
                                   f"{self.c.backend.dropped_images} input image(s) were dropped")
        if not all(v.passed for v in report.verify):
            report.status = "failed_checks"

    def _person_check(self, op: str, pid: int, face: Image.Image):
        v = self.c.verifier
        if op == "pose":
            return lambda im: v.check_question(f"pose_person_{pid}", im, self.spec.templates.pose_check_question)
        if op == "stylize":
            return lambda im: v.check_identity(f"identity_person_{pid}", face, im)
        return None

    def _edit_with_retries(self, step: Step, images: list[Image.Image], check, retries: int, path_stem: Path,
                           pid: int | None, report: RunReport, size: tuple[int, int] | None = None,
                           fallback: Image.Image | None = None) -> Image.Image:
        best: tuple[float, Image.Image] | None = None
        for attempt in range(retries + 1):
            seed = (step.seed or 0) + attempt * 7919
            t = time.perf_counter()
            kw: dict[str, Any] = {"negative_prompt": step.negative_prompt, "seed": seed}
            if size:
                kw.update(width=size[0], height=size[1])
            result = self.c.backend.edit(images, step.prompt, **kw)
            if size and result.size != size:
                result = result.resize(size, Image.LANCZOS)
            path = save(result, Path(f"{path_stem}_a{attempt}.png"))
            res: CheckResult | None = check(result) if check else None
            report.steps.append(StepRecord(person=pid, op=step.op, attempt=attempt, seed=seed, prompt=step.prompt,
                                           output=path, seconds=round(time.perf_counter() - t, 2), check=res))
            if res is None or res.passed is not False:
                return result
            score = res.score if res.score is not None else 0.0
            if best is None or score > best[0]:
                best = (score, result)
        label = f"person {pid} {step.op}" if pid is not None else step.op
        if fallback is not None:
            report.warnings.append(f"{label}: all {retries + 1} attempts failed checks; used previous image")
            return fallback
        report.warnings.append(f"{label}: all {retries + 1} attempts failed checks; kept best attempt")
        return best[1] if best else images[0]

    # ------------------------------------------------------------------ baseline mode

    def _run_single_prompt(self, mode: str, img: Image.Image, analysis: Analysis, out: Path, seed: int,
                           brief: str | None, report: RunReport) -> None:
        """mode = 'single:<backend>' e.g. single:main (the pipeline backend) or single:gemini."""
        if not brief:
            raise ValueError("single-prompt mode needs a brief (--brief file)")
        name = mode.split(":", 1)[1] if ":" in mode else "main"
        backend = self.c.backend if name == "main" else self.c.baseline_backends.get(name)
        if backend is None:
            raise ValueError(f"baseline backend '{name}' is not enabled in pipeline.yaml -> baselines")
        budget = budget_report(brief, self.cfg.get("planner.tokenizer"))
        report.prompt_tokens["brief"] = int(budget["tokens"])
        if budget.get("over_diffusers_default"):
            report.warnings.append(f"brief is ~{budget['tokens']} tokens; Qwen's default 512-token limit cuts it near: "
                                   f"'{budget.get('first_words_lost_at_512', '')}'")
        t = time.perf_counter()
        W, H = self.spec.canvas.width, self.spec.canvas.height
        result = backend.edit([img], brief, seed=seed, width=W, height=H)
        path = save(result, out / "baseline.png")
        report.outputs["baseline"] = path
        report.steps.append(StepRecord(person=None, op="edit", attempt=0, seed=seed, prompt=brief[:200] + "...",
                                       output=path, seconds=round(time.perf_counter() - t, 2)))
        report.verify.append(self.c.verifier.verify_free("baseline", result, analysis))
        if not all(v.passed for v in report.verify):
            report.status = "failed_checks"

    def _calls(self) -> int:
        return self.c.backend.calls + sum(b.calls for b in self.c.baseline_backends.values())


def _dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
