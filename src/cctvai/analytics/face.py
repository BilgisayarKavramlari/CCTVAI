"""Face analytics using DeepFace."""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from ..detectors.base import BoundingBox, FaceAnalytics

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from deepface import DeepFace  # type: ignore
except Exception:  # pragma: no cover
    DeepFace = None  # type: ignore


class DeepFaceAnalytics(FaceAnalytics):
    """Use DeepFace to extract age, gender and emotions."""

    def __init__(self) -> None:
        if DeepFace is None:
            raise RuntimeError(
                "DeepFace is required for face analytics. Install with `pip install deepface`."
            )
        LOGGER.info("DeepFace analytics initialised")

    def analyze(self, frame: np.ndarray, bbox: BoundingBox) -> Dict[str, object]:
        x1, y1, x2, y2 = map(int, bbox.as_tuple())
        crop = frame[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)]
        analysis = DeepFace.analyze(
            crop,
            actions=("age", "gender", "emotion"),
            enforce_detection=False,
            detector_backend="retinaface",
        )
        if isinstance(analysis, list):
            analysis = analysis[0]
        return {
            "age": analysis.get("age"),
            "gender": analysis.get("gender"),
            "emotion": analysis.get("dominant_emotion"),
            "emotions": analysis.get("emotion"),
        }


__all__ = ["DeepFaceAnalytics"]
