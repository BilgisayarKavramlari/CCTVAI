"""Behaviour classifier built on HuggingFace transformers."""
from __future__ import annotations

import logging
from collections import deque
from typing import Deque, List, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from transformers import pipeline
except Exception:  # pragma: no cover
    pipeline = None  # type: ignore

from .base import BehaviourClassifier

LOGGER = logging.getLogger(__name__)


class VideoBehaviourClassifier(BehaviourClassifier):
    """Wrap a HuggingFace video classification pipeline."""

    def __init__(self, model_name: str, window: int = 16) -> None:
        if pipeline is None:
            raise RuntimeError(
                "transformers is required for behaviour classification. Install with `pip install transformers torch`."
            )
        self._pipeline = pipeline("video-classification", model=model_name)
        self.window = window
        self._buffer: Deque[np.ndarray] = deque(maxlen=window)
        LOGGER.info("Loaded behaviour classifier %s", model_name)

    def update(self, frame: np.ndarray) -> Tuple[str, float]:
        self._buffer.append(frame)
        if len(self._buffer) < self.window:
            return ("unknown", 0.0)
        return self.predict(list(self._buffer))

    def predict(self, frames: List[np.ndarray]) -> Tuple[str, float]:
        if len(frames) < self.window:
            raise ValueError("Not enough frames provided")
        video = np.stack(frames, axis=0)
        outputs = self._pipeline(video)
        if isinstance(outputs, list):
            outputs = outputs[0]
        label = outputs[0]["label"]
        score = float(outputs[0]["score"])
        return (label, score)


__all__ = ["VideoBehaviourClassifier"]
