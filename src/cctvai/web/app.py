"""FastAPI web dashboard for CCTVAI."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..config import CCTVAIConfig
from ..storage import AlertLog, StreamStat

LOGGER = logging.getLogger(__name__)


DASHBOARD_HTML = Path(__file__).with_name("dashboard.html")


def create_app(config: CCTVAIConfig, session_factory) -> FastAPI:
    app = FastAPI(title="CCTVAI Dashboard")

    if DASHBOARD_HTML.exists():
        static_dir = DASHBOARD_HTML.parent
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def root():
        if DASHBOARD_HTML.exists():
            return HTMLResponse(DASHBOARD_HTML.read_text())
        return {"status": "ok", "message": "Dashboard assets missing"}

    @app.get("/api/streams")
    async def list_streams():
        return [stream.__dict__ for stream in config.streams]

    @app.get("/api/alerts")
    async def list_alerts(limit: int = 50):
        session = session_factory()
        try:
            alerts = (
                session.query(AlertLog)
                .order_by(AlertLog.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "stream_name": a.stream_name,
                    "event_type": a.event_type,
                    "confidence": a.confidence,
                    "message": a.message,
                    "created_at": a.created_at.isoformat(),
                }
                for a in alerts
            ]
        finally:
            session.close()

    @app.get("/api/stats")
    async def list_stats(limit: int = 50):
        session = session_factory()
        try:
            stats = (
                session.query(StreamStat)
                .order_by(StreamStat.captured_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "stream_name": s.stream_name,
                    "captured_at": s.captured_at.isoformat(),
                    "person_count": s.person_count,
                    "male_count": s.male_count,
                    "female_count": s.female_count,
                    "age_distribution": s.age_distribution,
                    "emotion_distribution": s.emotion_distribution,
                }
                for s in stats
            ]
        finally:
            session.close()

    return app


__all__ = ["create_app"]
