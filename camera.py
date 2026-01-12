"""
Camera capture thread.
"""

import collections
import logging
import threading
import time
from typing import Optional, Deque

import cv2
import numpy as np

from config import AppConfig

logger = logging.getLogger(__name__)


class CameraCapture(threading.Thread):
    def __init__(self, cfg: AppConfig):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.cap = None
        self.running = False
        # Circular buffer to store frames - max size to prevent memory issues
        # Set to 2x clip_seconds to ensure we have enough buffer
        max_buffer_size = max(100, cfg.clip_seconds * cfg.fps * 2)
        self.frame_buffer: Deque[np.ndarray] = collections.deque(maxlen=max_buffer_size)
        self.lock = threading.Lock()

    def run(self):
        logger.info(f"Starting capture from {self.cfg.camera_source}")
        self.cap = cv2.VideoCapture(self.cfg.camera_source)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Frame not read, sleeping briefly...")
                time.sleep(0.1)
                continue
            with self.lock:
                # Add new frame to the circular buffer
                self.frame_buffer.append(frame.copy())
                logger.debug(f"Frame added to buffer, size={len(self.frame_buffer)}")
            time.sleep(1.0 / self.cfg.fps)
        self.cap.release()
        logger.info("Camera capture stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Returns the oldest unread frame from the buffer (FIFO).
        Each frame is returned only once, ensuring no duplicates or skips.
        """
        with self.lock:
            if not self.frame_buffer:
                return None
            # Pop the oldest frame (first in, first out)
            frame = self.frame_buffer.popleft()
        # print("[camera] get_frame: returned frame, buffer size=", len(self.frame_buffer))
        return frame

    def stop(self):
        self.running = False

