"""Identity scoring: is the stylized rider still recognizably the same guest?

Scorers return a number in [0, 1] (higher = same person) or None if no face could be read.
Pick per style and CALIBRATE the threshold on your own data (see README -> Calibration):
  * insightface  - good for photoreal steps; drops sharply on cartoons. Non-commercial weights.
  * rekognition  - production-safe second opinion; also photo-trained.
  * vlm_judge    - pairwise "same person?" with a VLM; handles stylized faces better.
  * styleid      - hook for the StyleID style-robust encoder (arXiv 2604.21689); add when you test it.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from souvenir.config import read_prompt
from souvenir.imaging import to_jpeg_bytes
from souvenir.vlm import VLM


class IdentityScorer(Protocol):
    name: str

    def score(self, reference_face: Image.Image, candidate: Image.Image) -> float | None: ...


class InsightFaceScorer:
    name = "insightface"

    def __init__(self, model: str = "buffalo_l", ctx_id: int = 0):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name=model)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def _embed(self, img: Image.Image) -> np.ndarray | None:
        faces = self.app.get(np.asarray(img.convert("RGB"))[:, :, ::-1].copy())
        if not faces:
            return None
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return f.normed_embedding

    def score(self, reference_face: Image.Image, candidate: Image.Image) -> float | None:
        a, b = self._embed(reference_face), self._embed(candidate)
        if a is None or b is None:
            return None
        return float(np.clip(np.dot(a, b), -1, 1))


class RekognitionScorer:
    name = "rekognition"

    def __init__(self, region: str = "us-west-2"):
        import boto3

        self.client = boto3.client("rekognition", region_name=region)

    def score(self, reference_face: Image.Image, candidate: Image.Image) -> float | None:
        try:
            resp = self.client.compare_faces(
                SourceImage={"Bytes": to_jpeg_bytes(reference_face)},
                TargetImage={"Bytes": to_jpeg_bytes(candidate)},
                SimilarityThreshold=0,
            )
        except self.client.exceptions.InvalidParameterException:
            return None  # no face found in one of the images
        matches = resp.get("FaceMatches", [])
        if not matches:
            return 0.0
        return max(m["Similarity"] for m in matches) / 100.0


class VLMJudgeScorer:
    name = "vlm_judge"

    def __init__(self, vlm: VLM):
        self.vlm = vlm
        self.system = read_prompt("identity_judge.md")

    def score(self, reference_face: Image.Image, candidate: Image.Image) -> float | None:
        try:
            out = self.vlm.chat_json(self.system, "Image 1 is the real photo, image 2 the illustration.", [reference_face, candidate])
        except Exception:
            return None
        same = bool(out.get("same_person"))
        conf = float(out.get("confidence", 0.5))
        return conf if same else 1.0 - conf


class MockScorer:
    name = "mock"

    def score(self, reference_face: Image.Image, candidate: Image.Image) -> float | None:
        return 0.9


def build_scorer(section: dict[str, Any], vlm: VLM) -> IdentityScorer:
    kind = section.get("kind", "mock")
    if kind == "insightface":
        return InsightFaceScorer(model=section.get("model", "buffalo_l"))
    if kind == "rekognition":
        return RekognitionScorer(region=section.get("region", "us-west-2"))
    if kind == "vlm_judge":
        return VLMJudgeScorer(vlm)
    if kind == "styleid":
        raise NotImplementedError(
            "StyleID hook: download the released model from the StyleID project page, wrap it in a class "
            "with .score(reference_face, candidate) like InsightFaceScorer, and register it here."
        )
    if kind == "mock":
        return MockScorer()
    raise ValueError(f"unknown identity.kind: {kind}")


def match(scores: np.ndarray) -> list[tuple[int, int, float]]:
    """Hungarian assignment on a (references x candidates) similarity matrix. Returns (ref, cand, score)."""
    if scores.size == 0:
        return []
    filled = np.nan_to_num(scores, nan=-1.0)
    rows, cols = linear_sum_assignment(-filled)
    return [(int(r), int(c), float(scores[r, c])) for r, c in zip(rows, cols)]
