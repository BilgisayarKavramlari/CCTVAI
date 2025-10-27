"""Video stream manager."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Callable, Dict, Iterable, Iterator, Optional

import cv2
import numpy as np

from ..config import StreamConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class Frame:
    stream: StreamConfig
    frame_id: int
    data: np.ndarray
    timestamp: float


class StreamWorker(threading.Thread):
    """Read frames from a stream and push to queue."""

    def __init__(
        self,
        stream: StreamConfig,
        queue: Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True, name=f"stream-{stream.name}")
        self.stream = stream
        self.queue = queue
        self.stop_event = stop_event
        self._capture: Optional[cv2.VideoCapture] = None

    def open_capture(self) -> cv2.VideoCapture:
        source = int(self.stream.url) if self.stream.url.isdigit() else self.stream.url
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open stream {self.stream.name}: {self.stream.url}")
        LOGGER.info("Opened stream %s", self.stream.name)
        return capture

    def run(self) -> None:
        frame_id = 0
        self._capture = self.open_capture()
        while not self.stop_event.is_set():
            ok, frame = self._capture.read()
            if not ok:
                LOGGER.warning("Stream %s ended", self.stream.name)
                break
            frame_id += 1
            if frame_id % self.stream.sampling_rate != 0:
                continue
            self.queue.put(
                Frame(
                    stream=self.stream,
                    frame_id=frame_id,
                    data=frame,
                    timestamp=time.time(),
                )
            )
        if self._capture is not None:
            self._capture.release()
        LOGGER.info("Stream worker %s stopped", self.stream.name)


class StreamManager:
    """Manage multiple stream workers."""

    def __init__(self, streams: Iterable[StreamConfig]) -> None:
        self.streams = [s for s in streams if s.enabled]
        self.queue: Queue[Frame] = Queue(maxsize=32)
        self.stop_event = threading.Event()
        self.workers = [StreamWorker(stream=s, queue=self.queue, stop_event=self.stop_event) for s in self.streams]

    def start(self) -> None:
        for worker in self.workers:
            worker.start()
        LOGGER.info("Started %d stream workers", len(self.workers))

    def stop(self) -> None:
        self.stop_event.set()
        for worker in self.workers:
            worker.join(timeout=2)
        LOGGER.info("All stream workers stopped")

    def frames(self) -> Iterator[Frame]:
        while not self.stop_event.is_set():
            frame = self.queue.get()
            yield frame


__all__ = ["StreamManager", "Frame"]
