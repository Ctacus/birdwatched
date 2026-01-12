"""
Storage utilities for saving captured images and clips.
"""

import datetime
import os
from typing import List, Optional

import cv2
import numpy as np

from config import AppConfig


class StorageManager:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        os.makedirs(self.cfg.image_dir, exist_ok=True)
        os.makedirs(self.cfg.video_dir, exist_ok=True)

    @staticmethod
    def _timestamp() -> str:
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_image(self, frame, prefix: str = "bird") -> str:
        fname = f"{prefix}_{self._timestamp()}.jpg"
        path = os.path.join(self.cfg.image_dir, fname)
        cv2.imwrite(path, frame)
        return path

    def save_video(self, frames: List[np.ndarray], prefix: str = "clip") -> Optional[str]:
        if not frames:
            return None
        h, w = frames[0].shape[:2]
        fname = f"{prefix}_{self._timestamp()}.mp4"
        path = os.path.join(self.cfg.video_dir, fname)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # fourcc = cv2.VideoWriter_fourcc(*"avc1")  # или x264 или h264 -- огромный битрейт и не заливается в телегу -- отваливается по таймауту
        writer = cv2.VideoWriter(path, fourcc, self.cfg.fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        return path

