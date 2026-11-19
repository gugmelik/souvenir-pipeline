"""People detection: how many riders, and where each face and body is.

The count N is decided HERE, in code, and never left to the image model's prompt.
Small faces far behind the group are treated as bystanders (see `min_face_ratio`).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from PIL import Image

from souvenir.imaging import to_jpeg_bytes
from souvenir.schemas import Box, DetectedPerson


class FaceFound:
    def __init__(self, box: Box, score: float, embedding: np.ndarray | None = None):
        self.box = box
        self.score = score
        self.embedding = embedding


class Detector(Protocol):
    name: str

    def faces(self, img: Image.Image) -> list[FaceFound]: ...

    def people(self, img: Image.Image) -> tuple[list[DetectedPerson], int]: ...


def body_from_face(face: Box, width: int, height: int) -> Box:
    """Heuristic full-body box from a face box (used when no person box is available)."""
    fw, fh = face.w, face.h
    return Box(
        x0=face.cx - 2.2 * fw,
        y0=face.y0 - 0.7 * fh,
        x1=face.cx + 2.2 * fw,
        y1=face.y1 + 7.0 * fh,
    ).clip(width, height)


def select_riders(faces: list[FaceFound], min_face_ratio: float, det_threshold: float) -> tuple[list[FaceFound], int]:
    """Keep confident faces that are not much smaller than the largest one."""
    confident = [f for f in faces if f.score >= det_threshold]
    if not confident:
        return [], len(faces)
    largest = max(f.box.w for f in confident)
    riders = [f for f in confident if f.box.w >= min_face_ratio * largest]
    return riders, len(faces) - len(riders)


class _BaseDetector:
    name = "base"

    def __init__(self, min_face_ratio: float = 0.35, det_threshold: float = 0.5):
        self.min_face_ratio = min_face_ratio
        self.det_threshold = det_threshold

    def faces(self, img: Image.Image) -> list[FaceFound]:  # pragma: no cover - interface
        raise NotImplementedError

    def person_boxes(self, img: Image.Image) -> list[Box]:
        return []

    def people(self, img: Image.Image) -> tuple[list[DetectedPerson], int]:
        riders, ignored = select_riders(self.faces(img), self.min_face_ratio, self.det_threshold)
        riders.sort(key=lambda f: f.box.cx)  # left -> right, matches the family's arrangement
        bodies = self.person_boxes(img)
        people: list[DetectedPerson] = []
        for i, f in enumerate(riders, start=1):
            body = _body_containing(f.box, bodies) or body_from_face(f.box, img.width, img.height)
            # Tight groups: keep each body crop between the midpoints to the neighbours' faces,
            # so the pose step sees one person, not three. (Upgrade path: SAM 2 prompted with the face box.)
            idx = i - 1
            margin = 0.15 * f.box.w
            if idx > 0:
                body = body.model_copy(update={"x0": max(body.x0, (riders[idx - 1].box.cx + f.box.cx) / 2 - margin)})
            if idx < len(riders) - 1:
                body = body.model_copy(update={"x1": min(body.x1, (riders[idx + 1].box.cx + f.box.cx) / 2 + margin)})
            people.append(DetectedPerson(id=i, face_box=f.box, body_box=body, det_score=f.score))
        return people, ignored


def _body_containing(face: Box, bodies: list[Box]) -> Box | None:
    best, best_area = None, None
    for b in bodies:
        if b.x0 <= face.cx <= b.x1 and b.y0 <= face.cy <= b.y1:
            area = b.w * b.h
            if best_area is None or area < best_area:
                best, best_area = b, area
    return best


class InsightFaceDetector(_BaseDetector):
    """InsightFace buffalo_l. NOTE: the pretrained weights are licensed for non-commercial research."""

    name = "insightface"

    def __init__(self, model: str = "buffalo_l", ctx_id: int = 0, det_size: int = 640, **kw: Any):
        super().__init__(**kw)
        from insightface.app import FaceAnalysis  # pip install .[face]

        self.app = FaceAnalysis(name=model)
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    def faces(self, img: Image.Image) -> list[FaceFound]:
        bgr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()
        out = []
        for f in self.app.get(bgr):
            x0, y0, x1, y1 = [float(v) for v in f.bbox]
            out.append(FaceFound(Box(x0=x0, y0=y0, x1=x1, y1=y1), float(f.det_score), f.normed_embedding))
        return out


class RekognitionDetector(_BaseDetector):
    """AWS Rekognition (what occo already runs in production). Uses DetectFaces + DetectLabels(Person)."""

    name = "rekognition"

    def __init__(self, region: str = "us-west-2", **kw: Any):
        super().__init__(**kw)
        import boto3  # pip install .[aws]

        self.client = boto3.client("rekognition", region_name=region)

    def faces(self, img: Image.Image) -> list[FaceFound]:
        resp = self.client.detect_faces(Image={"Bytes": to_jpeg_bytes(img)}, Attributes=["DEFAULT"])
        out = []
        for d in resp.get("FaceDetails", []):
            bb = d["BoundingBox"]
            box = Box(
                x0=bb["Left"] * img.width,
                y0=bb["Top"] * img.height,
                x1=(bb["Left"] + bb["Width"]) * img.width,
                y1=(bb["Top"] + bb["Height"]) * img.height,
            )
            out.append(FaceFound(box, float(d.get("Confidence", 0)) / 100.0))
        return out

    def person_boxes(self, img: Image.Image) -> list[Box]:
        resp = self.client.detect_labels(Image={"Bytes": to_jpeg_bytes(img)}, MaxLabels=20, MinConfidence=60)
        boxes = []
        for label in resp.get("Labels", []):
            if label.get("Name") != "Person":
                continue
            for inst in label.get("Instances", []):
                bb = inst["BoundingBox"]
                boxes.append(
                    Box(
                        x0=bb["Left"] * img.width,
                        y0=bb["Top"] * img.height,
                        x1=(bb["Left"] + bb["Width"]) * img.width,
                        y1=(bb["Top"] + bb["Height"]) * img.height,
                    )
                )
        return boxes


class MockDetector(_BaseDetector):
    """Returns fixed face boxes (tests / demo). Boxes are (x0, y0, x1, y1) in pixels."""

    name = "mock"

    def __init__(self, boxes: list[tuple[float, float, float, float]] | None = None, **kw: Any):
        super().__init__(**kw)
        self.boxes = boxes or []

    def faces(self, img: Image.Image) -> list[FaceFound]:
        return [FaceFound(Box(x0=a, y0=b, x1=c, y1=d), 0.99) for a, b, c, d in self.boxes]


def build_detector(section: dict[str, Any], mock_boxes: list[tuple[float, float, float, float]] | None = None) -> Detector:
    kind = section.get("kind", "mock")
    common = dict(
        min_face_ratio=float(section.get("min_face_ratio", 0.35)),
        det_threshold=float(section.get("det_threshold", 0.5)),
    )
    if kind == "insightface":
        return InsightFaceDetector(model=section.get("model", "buffalo_l"), ctx_id=int(section.get("ctx_id", 0)), **common)
    if kind == "rekognition":
        return RekognitionDetector(region=section.get("region", "us-west-2"), **common)
    if kind == "mock":
        return MockDetector(boxes=mock_boxes, **common)
    raise ValueError(f"unknown detector.kind: {kind}")
