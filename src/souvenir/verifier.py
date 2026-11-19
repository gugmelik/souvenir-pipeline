"""Stage 6 - verifier: decide in code whether an output is sellable, and which step to retry.

Checks
  rider_count         VLM counts people == N; detector must not find MORE faces than N
  identity_per_rider  each rider (cropped by the compositor's layout box) vs that guest's face crop
  vlm_checklist       the product checklist + per-rider clothing questions, yes/no
  ocr_text            banned text absent, required text present
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from souvenir.config import read_prompt
from souvenir.detect import Detector
from souvenir.identity import IdentityScorer, match
from souvenir.imaging import crop, load_image
from souvenir.ocr import OCR, normalize
from souvenir.schemas import Analysis, Box, CheckResult, ProductSpec, VerifyReport
from souvenir.vlm import VLM


class Verifier:
    def __init__(self, spec: ProductSpec, vlm: VLM, scorer: IdentityScorer, detector: Detector, ocr: OCR,
                 identity_threshold: float, checklist_min_yes: float = 1.0):
        self.spec = spec
        self.vlm = vlm
        self.scorer = scorer
        self.detector = detector
        self.ocr = ocr
        self.identity_threshold = identity_threshold
        self.checklist_min_yes = checklist_min_yes
        self.checklist_system = read_prompt("verifier_checklist.md")
        self.count_system = read_prompt("verifier_count.md")

    # ------------------------------------------------------------------ single checks

    def yes_no(self, img: Image.Image, questions: list[str]) -> list[dict[str, Any]]:
        user = "Answer each question about the image:\n" + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        try:
            out = self.vlm.chat_json(self.checklist_system, user, [img])
            answers = out.get("answers", [])
        except Exception as exc:
            return [{"q": q, "answer": "unknown", "detail": str(exc)} for q in questions]
        return answers[: len(questions)]

    def check_question(self, name: str, img: Image.Image, question: str) -> CheckResult:
        ans = self.yes_no(img, [question])
        a = str(ans[0].get("answer", "unknown")).lower() if ans else "unknown"
        passed = True if a == "yes" else False if a == "no" else None
        return CheckResult(name=name, passed=passed, detail=f"{question} -> {a}")

    def check_identity(self, name: str, reference_face: Image.Image, candidate: Image.Image) -> CheckResult:
        s = self.scorer.score(reference_face, candidate)
        if s is None:
            return CheckResult(name=name, passed=None, detail=f"{self.scorer.name}: no face readable")
        return CheckResult(name=name, passed=s >= self.identity_threshold, score=s,
                           detail=f"{self.scorer.name} {s:.3f} vs threshold {self.identity_threshold}")

    def check_count(self, img: Image.Image, n: int) -> CheckResult:
        try:
            out = self.vlm.chat_json(self.count_system, "How many people are in this image?", [img])
            vlm_count = out.get("count")
        except Exception:
            vlm_count = None
        faces = len(self.detector.faces(img))
        detail = f"expected {n}, vlm={vlm_count}, detector faces={faces}"
        if vlm_count is not None and int(vlm_count) != n:
            return CheckResult(name="rider_count", passed=False, detail=detail)
        if faces > n:
            return CheckResult(name="rider_count", passed=False, detail=detail + " (extra faces)")
        passed = True if vlm_count is not None else None
        return CheckResult(name="rider_count", passed=passed, detail=detail)

    def check_text(self, img: Image.Image) -> CheckResult:
        texts = self.ocr.read(img)
        if not texts and self.ocr.name == "none":
            return CheckResult(name="ocr_text", passed=None, detail="OCR disabled")
        joined = normalize(" ".join(texts))
        banned = [b for b in self.spec.banned_text_ocr if normalize(b) in joined]
        missing = [r for r in self.spec.required_text if normalize(r) not in joined]
        if banned:
            return CheckResult(name="ocr_text", passed=False, detail=f"banned text found: {banned}; read={texts}")
        if missing:
            return CheckResult(name="ocr_text", passed=None if not texts else False, detail=f"required text missing: {missing}; read={texts}")
        return CheckResult(name="ocr_text", passed=True, detail=f"read={texts}")

    # ------------------------------------------------------------------ full reports

    def verify_composite(self, target: str, img: Image.Image, analysis: Analysis, boxes: dict[int, Box]) -> VerifyReport:
        """For pipeline outputs: we know where each rider is, so identity is checked per rider."""
        checks = [self.check_count(img, analysis.n_people)]
        cards = {c.id: c for c in analysis.cards}
        for pid, box in boxes.items():
            rider = crop(img, box.scaled(1.1))
            face_ref = load_image(analysis.crops[str(pid)]["face"])
            checks.append(self.check_identity(f"identity_rider_{pid}", face_ref, rider))
            card = cards.get(pid)
            if card and card.clothing:
                checks.append(self.check_question(f"clothing_rider_{pid}", rider, f"Is this rider wearing {card.clothing}?"))
        checks.append(self._checklist(img))
        checks.append(self.check_text(img))
        return VerifyReport(target=target, checks=checks)

    def verify_free(self, target: str, img: Image.Image, analysis: Analysis) -> VerifyReport:
        """For single-prompt baselines: find faces, match them to guests with the Hungarian algorithm."""
        checks = [self.check_count(img, analysis.n_people)]
        found = self.detector.faces(img)
        refs = [load_image(analysis.crops[str(p.id)]["face"]) for p in analysis.persons]
        cands = [crop(img, f.box.scaled(1.8).square()) for f in found]
        if refs and cands:
            m = np.full((len(refs), len(cands)), np.nan)
            for i, r in enumerate(refs):
                for j, c in enumerate(cands):
                    s = self.scorer.score(r, c)
                    m[i, j] = np.nan if s is None else s
            pairs = match(m)
            matched = {i for i, _, _ in pairs}
            for i, j, s in pairs:
                ok = None if np.isnan(s) else s >= self.identity_threshold
                checks.append(CheckResult(name=f"identity_rider_{analysis.persons[i].id}", passed=ok,
                                          score=None if np.isnan(s) else s, detail=f"matched output face {j}"))
            for i in set(range(len(refs))) - matched:
                checks.append(CheckResult(name=f"identity_rider_{analysis.persons[i].id}", passed=False, detail="no matching face in output"))
        else:
            checks.append(CheckResult(name="identity", passed=None, detail=f"{len(cands)} faces detected in output"))
        checks.append(self._checklist(img))
        checks.append(self.check_text(img))
        return VerifyReport(target=target, checks=checks)

    def _checklist(self, img: Image.Image) -> CheckResult:
        if not self.spec.checklist:
            return CheckResult(name="vlm_checklist", passed=None, detail="no checklist in spec")
        answers = self.yes_no(img, self.spec.checklist)
        yes = sum(1 for a in answers if str(a.get("answer", "")).lower() == "yes")
        known = sum(1 for a in answers if str(a.get("answer", "")).lower() in ("yes", "no"))
        if known == 0:
            return CheckResult(name="vlm_checklist", passed=None, detail="VLM gave no answers")
        frac = yes / len(self.spec.checklist)
        failed = [a.get("q", "") for a in answers if str(a.get("answer", "")).lower() == "no"]
        return CheckResult(name="vlm_checklist", passed=frac >= self.checklist_min_yes, score=frac,
                           detail=f"{yes}/{len(self.spec.checklist)} yes; failed: {failed}")
