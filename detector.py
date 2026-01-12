"""
Movement detector with simple state machine.
"""

import collections
import logging
import threading
import time
from typing import Deque
from xmlrpc.client import Error

import cv2
import numpy as np

from base_camera import BaseCameraCapture
from config import AppConfig
from notifiers import SoundNotifier, TelegramNotifier
from storage import StorageManager

logger = logging.getLogger(__name__)

class ClipBuffer:
    def __init__(self, fps: int, clip_seconds: int):
        self.max_clip_length = clip_seconds * fps
        self.buffer: Deque[np.ndarray] = collections.deque(maxlen=self.max_clip_length * 2)
        self.motion_flags: Deque[float] = collections.deque(maxlen=self.buffer.maxlen)
        if self.buffer.maxlen <= self.max_clip_length:
            raise Error("Buffer must be at least 1 frame wider than clip")
        self.window_totals: Deque[float] = collections.deque(maxlen=self.buffer.maxlen - self.max_clip_length + 1)
        self.window_totals.append(0) # initial window total



    def append(self, frame: np.ndarray, motion: float):
        if motion > 1 or motion < 0:
            logger.warning(f"Incorrect motion value: {motion}! (expected in range [0;1])")
            motion = 0
        self.buffer.append(frame)
        self.motion_flags.append(motion)
        if len(self.buffer) <= self.max_clip_length: # initial windows accumulation
            self.window_totals[0] += motion
        else:
            dropped_motion_idx = -self.max_clip_length - 1
            total_motion = self.window_totals[-1] + motion - self.motion_flags[dropped_motion_idx]
            self.window_totals.append(total_motion)

    def motion_percent(self) -> float:
        return max(self.window_totals) / self.max_clip_length

    def is_ready(self) -> bool:
        return len(self.buffer) == self.buffer.maxlen

    def get_clip(self):
        best_windows_motion = max(self.window_totals)
        idx = self.window_totals.index(best_windows_motion)
        return list(self.buffer)[idx:idx + self.max_clip_length]




class Detector(threading.Thread):
    """
    Детектор с машиной состояний:
    IDLE -> TRIGGERED -> COOLDOWN -> IDLE -> ...
    """

    STATE_IDLE = "idle"
    STATE_TRIGGERED = "triggered"
    STATE_COOLDOWN = "cooldown"

    def __init__(
        self,
        cfg: AppConfig,
        camera: BaseCameraCapture,
        storage: StorageManager,
        telegram: TelegramNotifier,
        sound: SoundNotifier,
    ):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.camera = camera
        self.storage = storage
        self.telegram = telegram
        self.sound = sound

        self.bg = cv2.createBackgroundSubtractorKNN(
            history=200,
            dist2Threshold=1000,
            detectShadows=False,
        )

        self.state = self.STATE_IDLE
        self.trigger_counter = 0
        # self.buffer: Deque[np.ndarray] = collections.deque(maxlen=self.cfg.clip_seconds * self.cfg.fps)
        self.buffer: ClipBuffer = ClipBuffer(self.cfg.fps, self.cfg.clip_seconds)
        self.lock = threading.Lock()
        self.cooldown_start = 0.0
        self.is_recording = False

    # --------------------------
    def _detect_movement(self, frame: np.ndarray) -> bool:
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        mask = self.bg.apply(small)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) >= self.cfg.min_contour_area:
                return True
        return False

    # --------------------------
    def run(self):
        logger.info("Detector started")
        last_state = None
        last_movement = False
        last_counter = 0
        try:
            while True:
                time.sleep(1.0 / self.cfg.fps)
                # logger.debug("Fetching frame...")
                frame = self.camera.get_frame()
                if frame is None:
                    logger.debug("No frame available")
                    time.sleep(1.0 / self.cfg.fps / 2)
                    continue
                movement = self._detect_movement(frame)
                self.buffer.append(frame, movement)
                if self.state !=last_state or movement != last_movement or last_counter != self.trigger_counter:
                    logger.debug(f"state={self.state} movement={movement} counter={self.trigger_counter}")
                last_state, last_movement, last_counter = self.state, movement, self.trigger_counter
                if self.state == self.STATE_IDLE:
                    self._handle_idle(movement, frame)
                    continue

                if self.state == self.STATE_TRIGGERED:
                    logger.info(f"Cooldown started for {self.cfg.cooldown_seconds} seconds")
                    self.state = self.STATE_COOLDOWN
                    self.cooldown_start = time.time()
                    continue

                if self.state == self.STATE_COOLDOWN:
                    self._handle_cooldown()
                    continue

        except Exception as e:
            logger.error(f"Detector error: {e}", exc_info=True)
            time.sleep(1)

    def _handle_idle(self, movement: bool, frame: np.ndarray):
        if movement:
            self.trigger_counter += 1
        else:
            self.trigger_counter = max(0, self.trigger_counter - 1)

        if self.trigger_counter >= self.cfg.detection_frames_required and self.buffer.is_ready():
            motion_pct = self.buffer.motion_percent()
            if motion_pct >= self.cfg.movement_level_required:
                logging.info(f"Motion detected, level: {motion_pct}")
                self._trigger_event(frame)
                self.state = self.STATE_TRIGGERED
                self.trigger_counter = 0

    def _handle_cooldown(self):
        if time.time() - self.cooldown_start >= self.cfg.cooldown_seconds:
            logger.info("Cooldown ended")
            self.state = self.STATE_IDLE
            self.trigger_counter = 0

    # --------------------------
    def _trigger_event(self, frame: np.ndarray):
        logger.info("BIRD EVENT TRIGGERED")

        image_path = self.storage.save_image(frame, prefix="bird")

        threading.Thread(
            target=self.telegram.send_photo,
            args=(image_path, "Птица у кормушки!"),
            daemon=True,
        ).start()

        # NOTE: отключено из-за подвисания, вернуть при необходимости
        # threading.Thread(target=self.sound.play_async, daemon=True).start()

        if self.is_recording:
            logger.warning("Clip already recording, skip new clip")
            return

        threading.Thread(target=self._write_clip, daemon=True).start()

    # --------------------------
    def _write_clip(self):
        logger.info("Recording clip...")

        self.is_recording = True
        try:
            logger.info(f"Fetching clip from buffer...")
            frames = self.buffer.get_clip()
            path = self.storage.save_video(frames, prefix="birdclip")
            logger.info(f"Clip saved: {path}")

            threading.Thread(
                target=self.telegram.send_video,
                args=(path,),
                daemon=True,
            ).start()
        finally:
            self.is_recording = False

