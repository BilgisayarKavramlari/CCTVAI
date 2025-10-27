"""Command line interface for CCTVAI."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from .config import CCTVAIConfig, default_config
from .pipeline import CCTVAI
from .storage import AlertLog, StreamStat, create_storage
from .web.app import create_app

app = typer.Typer(add_completion=False)
console = Console()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def load_config(config_path: Optional[Path]) -> CCTVAIConfig:
    if config_path is None:
        console.print("[yellow]No config provided. Using defaults.[/yellow]")
        return default_config()
    import yaml

    data = yaml.safe_load(Path(config_path).read_text())
    # Basic manual mapping to dataclasses for brevity
    cfg = default_config()
    if "streams" in data:
        cfg.streams = [
            type(cfg.streams[0])(
                name=item["name"],
                url=str(item["url"]),
                enabled=item.get("enabled", True),
                sampling_rate=item.get("sampling_rate", 1),
            )
            for item in data["streams"]
        ]
    if "detection" in data:
        det = data["detection"]
        cfg.detection.person_detector = det.get("person_detector", cfg.detection.person_detector)
        cfg.detection.behaviour_model = det.get("behaviour_model", cfg.detection.behaviour_model)
    if "analytics" in data:
        cfg.analytics.aggregation_interval_seconds = data["analytics"].get(
            "aggregation_interval_seconds", cfg.analytics.aggregation_interval_seconds
        )
    if "storage" in data:
        cfg.storage.sqlite_path = Path(data["storage"].get("sqlite_path", cfg.storage.sqlite_path))
    return cfg


@app.command()
def run(config: Optional[Path] = typer.Option(None, help="Path to YAML configuration")):
    """Run the CCTVAI processing pipeline."""

    cfg = load_config(config)
    engine_factory = create_storage(cfg.storage)
    system = CCTVAI(cfg)
    try:
        system.start()
    except KeyboardInterrupt:
        console.print("\n[bold red]Stopping CCTVAI...[/bold red]")
        system.stop()
    finally:
        del engine_factory


@app.command()
def dashboard(config: Optional[Path] = typer.Option(None, help="Path to YAML configuration")):
    """Launch the FastAPI dashboard only."""

    cfg = load_config(config)
    session_factory = create_storage(cfg.storage)
    api = create_app(cfg, session_factory)
    uvicorn.run(api, host=cfg.web.host, port=cfg.web.port, reload=cfg.web.reload)


@app.command()
def stats(
    config: Optional[Path] = typer.Option(None, help="Path to YAML configuration"),
    limit: int = typer.Option(10, help="Number of rows to show"),
):
    """Display recent stats in the terminal."""

    cfg = load_config(config)
    session_factory = create_storage(cfg.storage)
    session = session_factory()
    try:
        table = Table("Kamera", "Zaman", "Kişi", "Erkek", "Kadın")
        for stat in (
            session.query(StreamStat)
            .order_by(StreamStat.captured_at.desc())
            .limit(limit)
            .all()
        ):
            table.add_row(
                stat.stream_name,
                stat.captured_at.isoformat(),
                str(stat.person_count or "-"),
                str(stat.male_count or "-"),
                str(stat.female_count or "-"),
            )
        console.print(table)
    finally:
        session.close()


@app.command()
def alerts(
    config: Optional[Path] = typer.Option(None, help="Path to YAML configuration"),
    limit: int = typer.Option(10, help="Number of rows to show"),
):
    """Display recent alerts."""

    cfg = load_config(config)
    session_factory = create_storage(cfg.storage)
    session = session_factory()
    try:
        table = Table("Kamera", "Olay", "Güven", "Mesaj", "Zaman")
        for alert in (
            session.query(AlertLog)
            .order_by(AlertLog.created_at.desc())
            .limit(limit)
            .all()
        ):
            table.add_row(
                alert.stream_name,
                alert.event_type,
                f"{alert.confidence:.2f}",
                alert.message,
                alert.created_at.isoformat(),
            )
        console.print(table)
    finally:
        session.close()


def main():
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
