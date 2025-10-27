"""SQLite storage helpers for CCTVAI."""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Iterable, List, Optional

from sqlalchemy import Column, DateTime, Float, Integer, JSON, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import StorageConfig

LOGGER = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class StreamStat(Base):
    __tablename__ = "stream_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stream_name: Mapped[str] = mapped_column(String, index=True)
    captured_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    person_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    male_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    female_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    age_distribution: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    emotion_distribution: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(String, nullable=True)


class AlertLog(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stream_name: Mapped[str] = mapped_column(String, index=True)
    event_type: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float)
    message: Mapped[str] = mapped_column(String)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


def create_storage(cfg: StorageConfig):
    Path(cfg.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{cfg.sqlite_path}")
    if cfg.recreate:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    LOGGER.info("Connected to SQLite at %s", cfg.sqlite_path)
    return Session


def record_stat(
    session_factory,
    stream_name: str,
    person_count: Optional[int],
    male_count: Optional[int],
    female_count: Optional[int],
    age_distribution: Optional[dict],
    emotion_distribution: Optional[dict],
    notes: Optional[str] = None,
) -> None:
    session = session_factory()
    try:
        stat = StreamStat(
            stream_name=stream_name,
            person_count=person_count,
            male_count=male_count,
            female_count=female_count,
            age_distribution=age_distribution,
            emotion_distribution=emotion_distribution,
            notes=notes,
        )
        session.add(stat)
        session.commit()
    finally:
        session.close()


def record_alert(session_factory, stream_name: str, event_type: str, confidence: float, message: str) -> None:
    session = session_factory()
    try:
        entry = AlertLog(
            stream_name=stream_name,
            event_type=event_type,
            confidence=confidence,
            message=message,
        )
        session.add(entry)
        session.commit()
    finally:
        session.close()


__all__ = [
    "create_storage",
    "record_stat",
    "record_alert",
    "StreamStat",
    "AlertLog",
]
