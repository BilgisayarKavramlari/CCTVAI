"""Configuration models for CCTVAI pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class StreamConfig:
    """Represents a single video stream entry.

    Attributes:
        name: Friendly name for the stream. Used for UI and logs.
        url: OpenCV compatible source (device id, file path, RTSP/HTTP URL).
        enabled: Whether the stream should be processed.
        sampling_rate: Process every Nth frame to reduce load.
    """

    name: str
    url: str
    enabled: bool = True
    sampling_rate: int = 1


@dataclass
class BehaviourLabel:
    """Metadata describing a behaviour to detect."""

    name: str
    description: str
    positive_examples: List[Path] = field(default_factory=list)
    negative_examples: List[Path] = field(default_factory=list)


@dataclass
class DetectionConfig:
    """Configures the detection modules used in the pipeline."""

    person_detector: str = "yolov8n.pt"
    face_detector: str = "retinaface"
    behaviour_model: str = "MCG-NJU/videomae-base-finetuned-kinetics"
    behaviour_labels: Dict[str, BehaviourLabel] = field(default_factory=dict)


@dataclass
class AnalyticsConfig:
    """Metadata capture options."""

    collect_person_counts: bool = True
    collect_demographics: bool = True
    collect_psychometrics: bool = True
    aggregation_interval_seconds: int = 600


@dataclass
class StorageConfig:
    """Storage layer configuration."""

    sqlite_path: Path = Path("data/cctvai.db")
    recreate: bool = False


@dataclass
class WebConfig:
    """Settings for the optional FastAPI dashboard."""

    host: str = "0.0.0.0"
    port: int = 8080
    reload: bool = False


@dataclass
class CCTVAIConfig:
    """Top level configuration for the CCTVAI framework."""

    streams: List[StreamConfig] = field(default_factory=list)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    web: WebConfig = field(default_factory=WebConfig)


DEFAULT_STREAM = StreamConfig(name="Webcam", url="0")


def default_config() -> CCTVAIConfig:
    """Produce a usable default configuration."""

    cfg = CCTVAIConfig(streams=[DEFAULT_STREAM])
    cfg.detection.behaviour_labels = {
        "shoplifting": BehaviourLabel(
            name="shoplifting",
            description="Attempt to steal merchandise by concealing it.",
        ),
        "fainting": BehaviourLabel(
            name="fainting",
            description="Person collapsing to the floor.",
        ),
        "smoking": BehaviourLabel(
            name="smoking",
            description="Person holding or lighting a cigarette.",
        ),
        "lost_child": BehaviourLabel(
            name="lost_child",
            description="Unattended child appearing distressed.",
        ),
        "accident": BehaviourLabel(
            name="accident",
            description="Sudden collision or fall between individuals.",
        ),
    }
    return cfg


__all__ = [
    "StreamConfig",
    "BehaviourLabel",
    "DetectionConfig",
    "AnalyticsConfig",
    "StorageConfig",
    "WebConfig",
    "CCTVAIConfig",
    "default_config",
]
