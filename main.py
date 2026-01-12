"""
birdwatcher_mvp.py
MVP: монитор кормушки — детекция (MOG2/KNN), сохранение фото/видео,
отправка фото в Telegram, воспроизведение звука.
Настройки через environment variables или .env (см. пример ниже).
"""

import collections
import datetime
import os
import threading
import time
from dataclasses import dataclass
from typing import Deque, List, Optional

import cv2
import numpy as np
import requests
import simpleaudio as sa
from dotenv import load_dotenv


# ---------- Config & helpers ----------
@dataclass
class AppConfig:
    camera_source: int | str
    image_dir: str
    video_dir: str
    telegram_bot_token: str
    telegram_chat_id: str
    alert_sound_path: str
    min_contour_area: int
    detection_frames_required: int
    clip_seconds: int
    fps: int
    cooldown_seconds: int = 3

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv()  # optional .env support
        return cls(
            camera_source=cls._parse_camera_source(os.getenv("CAMERA_SOURCE", "0")),
            image_dir=os.getenv("IMAGE_DIR", "./data/images"),
            video_dir=os.getenv("VIDEO_DIR", "./data/videos"),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            alert_sound_path=os.getenv("ALERT_SOUND_PATH", "alert.wav"),
            min_contour_area=int(os.getenv("MIN_CONTOUR_AREA", "400")),
            detection_frames_required=int(os.getenv("DETECTION_FRAMES_REQUIRED", "3")),
            clip_seconds=int(os.getenv("CLIP_SECONDS", "6")),
            fps=int(os.getenv("FPS", "15")),
        )

    @staticmethod
    def _parse_camera_source(value: str) -> int | str:
        try:
            return int(value)
        except ValueError:
            return value


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
        writer = cv2.VideoWriter(path, fourcc, self.cfg.fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        return path


# ---------- Notifiers ----------
class TelegramNotifier:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def send_photo(self, photo_path: str, caption: Optional[str] = None) -> bool:
        if not self.cfg.telegram_bot_token or not self.cfg.telegram_chat_id:
            print("[telegram] token/chat_id not set, skipping send.")
            return False
        url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendPhoto"
        with open(photo_path, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": self.cfg.telegram_chat_id}
            if caption:
                data["caption"] = caption
            try:
                resp = requests.post(url, data=data, files=files, timeout=15)
                resp.raise_for_status()
                print("[telegram] photo sent:", photo_path)
                return True
            except Exception as e:
                print("[telegram] failed to send:", e)
                return False

    def send_video(self, video_path: str, caption: Optional[str] = None) -> bool:
        if not self.cfg.telegram_bot_token or not self.cfg.telegram_chat_id:
            print("[telegram] token/chat_id not set, skipping send.")
            return False
        try:
            url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendVideo"
            with open(video_path, "rb") as vf:
                files = {"video": vf}
                data = {"chat_id": self.cfg.telegram_chat_id, "caption": caption or "Клип с кормушки"}
                resp = requests.post(url, data=data, files=files, timeout=30)
                resp.raise_for_status()
                print("[telegram] clip sent")
                return True
        except Exception as e:
            print("[telegram] failed to send clip:", e)
            return False


class SoundNotifier:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def play_async(self) -> bool:
        try:
            wave_obj = sa.WaveObject.from_wave_file(self.cfg.alert_sound_path)
            wave_obj.play()
            return True
        except Exception as e:
            print("[sound] failed to play:", e)
            return False


# ---------- Camera capture thread ----------
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
            print("[camera] read frame...  ", end="")
            ret, frame = self.cap.read()
            if not ret:
                print("\n[camera] frame not read, sleeping briefly...")
                time.sleep(0.1)
                continue
            with self.lock:
                self.frame = frame.copy()
                print("frame copied  ")
            time.sleep(1.0 / self.cfg.fps)
        self.cap.release()
        print("[camera] stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            frame = self.frame
        if frame is None:
            return None
        print("[camera] get_frame copy")
        return np.copy(frame)

    def stop(self):
        self.running = False


# ---------- Detector thread ----------
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

    # --------------------------
    def _detect_movement(self, frame: np.ndarray) -> bool:
        print("|", end="")
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        print("|", end="")
        mask = self.bg.apply(small)
        print("|", end="")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        print("|", end="")
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        print("|", end="")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("|", end="")
        for c in contours:
            if cv2.contourArea(c) >= self.cfg.min_contour_area:
                return True
        return False

    # --------------------------
    def run(self):
        print("[detector] started")

        try:
            while True:
                print("[detector] fetching frame...")
                frame = self.camera.get_frame()

                if frame is None:
                    time.sleep(0.05)
                    continue
                print("[detector] frame fetched...")
                self.buffer.append(frame)
                print("[detector] buffer appended", end="")
                movement = self._detect_movement(frame)
                print()
                print(f"[detector] state={self.state} movement={movement} counter={self.trigger_counter}")

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

        threading.Thread(target=self._write_clip, daemon=True).start()

    # --------------------------
    def _write_clip(self):
        print("[detector] recording clip...")

        with self.lock:
            frames = list(self.buffer)

        extra = self.cfg.clip_seconds * self.cfg.fps
        for _ in range(extra):
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

        with self.lock:
            self.buffer.clear()


# ---------- Application ----------
class BirdWatcherApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.storage = StorageManager(cfg)
        self.telegram = TelegramNotifier(cfg)
        self.sound = SoundNotifier(cfg)
        self.camera = CameraCapture(cfg)
        self.detector = Detector(cfg, self.camera, self.storage, self.telegram, self.sound)

    def start(self):
        self.camera.start()
        self.detector.start()
        print("MVP running. Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping...")
            self.camera.stop()
            time.sleep(0.5)


def main():
    cfg = AppConfig.from_env()
    app = BirdWatcherApp(cfg)
    app.start()


if __name__ == "__main__":
    main()
