"""YOLO based person detector."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

import numpy as np

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    YOLO = None  # type: ignore

from .base import BoundingBox, Detection, PersonDetector

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _load_model(weights: str) -> YOLO:
    if YOLO is None:
        raise RuntimeError(
            "Ultralytics YOLO is not installed. Install with `pip install ultralytics`."
        )
    LOGGER.info("Loading YOLO weights %s", weights)
    return YOLO(weights)


class YoloPersonDetector(PersonDetector):
    """Detect persons using YOLOv8 models."""

    def __init__(self, weights: str = "yolov8n.pt", confidence: float = 0.35) -> None:
        self.weights = weights
        self.confidence = confidence
        self._model = _load_model(weights)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self._model.predict(frame, conf=self.confidence, classes=[0])
        detections: List[Detection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append(
                    Detection(
                        bbox=BoundingBox(
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x2),
                            y2=float(y2),
                            confidence=conf,
                            label="person",
                        )
                    )
                )
        return detections


__all__ = ["YoloPersonDetector"]
