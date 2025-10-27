"""Detector abstractions used by the CCTVAI pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol, Tuple

import numpy as np


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class Detection:
    """Standard detection payload returned by detectors."""

    bbox: BoundingBox
    embedding: np.ndarray | None = None


class FrameDetections(Protocol):
    """Protocol describing detection results for a frame."""

    frame_id: int
    detections: Iterable[Detection]


class PersonDetector(Protocol):
    """Detect persons within a frame."""

    def detect(self, frame: np.ndarray) -> List[Detection]:
        ...


class FaceAnalytics(Protocol):
    """Extract age/gender/emotion data from a face crop."""

    def analyze(self, frame: np.ndarray, bbox: BoundingBox) -> dict:
        ...


class BehaviourClassifier(Protocol):
    """Classify short video clips."""

    window: int

    def predict(self, frames: List[np.ndarray]) -> Tuple[str, float]:
        ...
