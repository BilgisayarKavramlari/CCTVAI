"""Core CCTVAI processing pipeline."""
from __future__ import annotations

import collections
import datetime as dt
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from .analytics.face import DeepFaceAnalytics
from .config import CCTVAIConfig, StreamConfig
from .detectors.behaviour import VideoBehaviourClassifier
from .detectors.base import BoundingBox, Detection
from .detectors.yolo import YoloPersonDetector
from .storage import create_storage, record_alert, record_stat
from .streaming.manager import Frame, StreamManager

LOGGER = logging.getLogger(__name__)


@dataclass
class PersonObservation:
    bbox: BoundingBox
    age: Optional[int] = None
    gender: Optional[str] = None
    emotion: Optional[str] = None
    emotions: Dict[str, float] = field(default_factory=dict)
    last_event: Optional[str] = None
    last_event_confidence: float = 0.0


@dataclass
class StreamState:
    stream: StreamConfig
    persons: List[PersonObservation] = field(default_factory=list)
    last_stat_flush: dt.datetime = field(default_factory=dt.datetime.utcnow)
    active_alerts: Dict[str, dt.datetime] = field(default_factory=dict)


class CCTVAI:
    """Top level orchestrator for the framework."""

    def __init__(self, config: CCTVAIConfig) -> None:
        self.config = config
        self.person_detector = YoloPersonDetector(config.detection.person_detector)
        self.face_analytics = DeepFaceAnalytics() if config.analytics.collect_demographics else None
        self.behaviour_classifier = VideoBehaviourClassifier(config.detection.behaviour_model)
        self.stream_manager = StreamManager(config.streams)
        self.storage = create_storage(config.storage)
        self.states: Dict[str, StreamState] = {s.name: StreamState(stream=s) for s in config.streams}
        self._stop_event = threading.Event()

    def start(self) -> None:
        LOGGER.info("Starting CCTVAI")
        self.stream_manager.start()
        try:
            for frame in self.stream_manager.frames():
                if self._stop_event.is_set():
                    break
                self._process_frame(frame)
        finally:
            self.stream_manager.stop()

    def stop(self) -> None:
        self._stop_event.set()

    def _process_frame(self, frame: Frame) -> None:
        LOGGER.debug("Processing frame %s:%d", frame.stream.name, frame.frame_id)
        detections = self.person_detector.detect(frame.data)
        observations: List[PersonObservation] = []
        for detection in detections:
            obs = PersonObservation(bbox=detection.bbox)
            if self.face_analytics:
                try:
                    face_data = self.face_analytics.analyze(frame.data, detection.bbox)
                    obs.age = face_data.get("age")
                    obs.gender = face_data.get("gender")
                    obs.emotion = face_data.get("emotion")
                    obs.emotions = face_data.get("emotions", {})
                except Exception as exc:  # pragma: no cover - analytics optional
                    LOGGER.exception("Face analytics failed: %s", exc)
            label, confidence = self.behaviour_classifier.update(frame.data)
            obs.last_event = label
            obs.last_event_confidence = confidence
            if label in self.config.detection.behaviour_labels and confidence > 0.6:
                record_alert(
                    self.storage,
                    stream_name=frame.stream.name,
                    event_type=label,
                    confidence=confidence,
                    message=f"Detected {label} with confidence {confidence:.2f}",
                )
                self.states[frame.stream.name].active_alerts[label] = dt.datetime.utcnow()
            observations.append(obs)
        state = self.states[frame.stream.name]
        state.persons = observations
        now = dt.datetime.utcnow()
        elapsed = (now - state.last_stat_flush).total_seconds()
        if elapsed >= self.config.analytics.aggregation_interval_seconds:
            self._flush_stats(frame.stream, state)
            state.last_stat_flush = now

    def _flush_stats(self, stream: StreamConfig, state: StreamState) -> None:
        LOGGER.info("Flushing stats for %s", stream.name)
        person_count = len(state.persons)
        gender_counts = collections.Counter(obs.gender for obs in state.persons if obs.gender)
        emotion_counts = collections.Counter(obs.emotion for obs in state.persons if obs.emotion)
        age_histogram: Dict[str, int] = collections.Counter()
        for obs in state.persons:
            if obs.age is not None:
                bucket = f"{(obs.age // 10) * 10}s"
                age_histogram[bucket] += 1
        record_stat(
            self.storage,
            stream_name=stream.name,
            person_count=person_count,
            male_count=gender_counts.get("Man") or gender_counts.get("Male"),
            female_count=gender_counts.get("Woman") or gender_counts.get("Female"),
            age_distribution=dict(age_histogram) if age_histogram else None,
            emotion_distribution=dict(emotion_counts) if emotion_counts else None,
            notes=None,
        )


__all__ = ["CCTVAI"]
