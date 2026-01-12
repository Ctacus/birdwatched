"""
Camera capture thread.
"""

import threading
import time
from typing import Optional

import cv2
import numpy as np

from config import AppConfig


class CameraCapture(threading.Thread):
    def __init__(self, cfg: AppConfig):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.cap = None
        self.running = False
        self.frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()

    def run(self):
        print("[camera] starting capture from", self.cfg.camera_source)
        self.cap = cv2.VideoCapture(self.cfg.camera_source)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        self.running = True
        while self.running:
            # print("[camera] read frame...  ", end="")
            ret, frame = self.cap.read()
            if not ret:
                print("\n[camera] frame not read, sleeping briefly...")
                time.sleep(0.1)
                continue
            with self.lock:
                self.frame = frame.copy()
                # print("frame copied  ")
            time.sleep(1.0 / self.cfg.fps)
        self.cap.release()
        print("[camera] stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            frame = self.frame
        if frame is None:
            return None
        # print("[camera] get_frame copy")
        return np.copy(frame)

    def stop(self):
        self.running = False

