"""
Movement detector with simple state machine.
"""

import collections
import threading
import time
from typing import Deque

import cv2
import numpy as np

from camera import CameraCapture
from config import AppConfig
from notifiers import SoundNotifier, TelegramNotifier
from storage import StorageManager


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
        camera: CameraCapture,
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
        self.buffer: Deque[np.ndarray] = collections.deque(maxlen=self.cfg.clip_seconds * self.cfg.fps)
        self.lock = threading.Lock()
        self.cooldown_start = 0.0
        self.is_recording = False

    # --------------------------
    def _detect_movement(self, frame: np.ndarray) -> bool:
        # print("|", end="")
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        # print("|", end="")
        mask = self.bg.apply(small)
        # print("|", end="")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # print("|", end="")
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # print("|", end="")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print("|", end="")
        for c in contours:
            if cv2.contourArea(c) >= self.cfg.min_contour_area:
                return True
        return False

    # --------------------------
    def run(self):
        print("[detector] started")
        last_state = None
        last_movement = False
        last_counter = 0
        try:
            while True:
                # print("[detector] fetching frame...")
                frame = self.camera.get_frame()

                if frame is None:
                    time.sleep(0.05)
                    continue
                # print("[detector] frame fetched...")
                self.buffer.append(frame)
                # print("[detector] buffer appended", end="")
                movement = self._detect_movement(frame)
                # print()
                if self.state !=last_state or movement != last_movement:
                    print(f"[detector] state={self.state} movement={movement} counter={self.trigger_counter}")
                last_state, last_movement, last_counter = self.state, movement, self.trigger_counter
                time.sleep(0.2)
                if self.state == self.STATE_IDLE:
                    self._handle_idle(movement, frame)
                    continue

                if self.state == self.STATE_TRIGGERED:
                    print("[detector] cooldown started")
                    self.state = self.STATE_COOLDOWN
                    self.cooldown_start = time.time()
                    continue

                if self.state == self.STATE_COOLDOWN:
                    self._handle_cooldown()
                    continue

                time.sleep(1.0 / self.cfg.fps)
        except Exception as e:
            print("[detector] ERROR:", e)
            import traceback

            traceback.print_exc()
            time.sleep(0.5)

    def _handle_idle(self, movement: bool, frame: np.ndarray):
        if movement:
            self.trigger_counter += 1
        else:
            self.trigger_counter = max(0, self.trigger_counter - 1)

        if self.trigger_counter >= self.cfg.detection_frames_required:
            self._trigger_event(frame)
            print("[detector] TRIGGERED!")
            self.state = self.STATE_TRIGGERED
            self.trigger_counter = 0

    def _handle_cooldown(self):
        if time.time() - self.cooldown_start >= self.cfg.cooldown_seconds:
            print("[detector] cooldown ended")
            self.state = self.STATE_IDLE
            self.trigger_counter = 0

    # --------------------------
    def _trigger_event(self, frame: np.ndarray):
        print("[detector] BIRD EVENT TRIGGERED")

        image_path = self.storage.save_image(frame, prefix="bird")

        threading.Thread(
            target=self.telegram.send_photo,
            args=(image_path, "Птица у кормушки!"),
            daemon=True,
        ).start()

        # NOTE: отключено из-за подвисания, вернуть при необходимости
        # threading.Thread(target=self.sound.play_async, daemon=True).start()

        if self.is_recording:
            print("[detector] clip already recording, skip new clip")
            return

        threading.Thread(target=self._write_clip, daemon=True).start()

    # --------------------------
    def _write_clip(self):
        print("[detector] recording clip...")

        self.is_recording = True
        try:
            target_frames = self.cfg.clip_seconds * self.cfg.fps
            with self.lock:
                # keep only the most recent frames up to the target clip length
                frames = list(self.buffer)[-target_frames:]
                # start fresh for the next potential trigger while we record this one
                self.buffer.clear()

            # capture only the remaining frames needed to reach target length
            remaining = max(0, target_frames - len(frames))
            for _ in range(remaining):
                f = self.camera.get_frame()
                if f is not None:
                    frames.append(f)
                time.sleep(1.0 / self.cfg.fps)

            path = self.storage.save_video(frames, prefix="birdclip")
            print("[detector] clip saved:", path)

            threading.Thread(
                target=self.telegram.send_video,
                args=(path,),
                daemon=True,
            ).start()
        finally:
            self.is_recording = False

