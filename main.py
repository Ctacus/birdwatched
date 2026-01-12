"""
birdwatcher_mvp.py
MVP: монитор кормушки — детекция (MOG2), сохранение фото/видео, отправка фото в Telegram, воспроизведение звука.
Настройки через environment variables или .env (см. пример ниже).
"""

import cv2
import time
import threading
import collections
import os
import requests
import datetime
import numpy as np
import simpleaudio as sa
from dotenv import load_dotenv

load_dotenv()  # optional .env support

# ---------- CONFIG ----------
CAMERA_SOURCE = int(os.getenv("CAMERA_SOURCE", "0"))  # 0 = webcam; or RTSP url as string
IMAGE_DIR = os.getenv("IMAGE_DIR", "./data/images")
VIDEO_DIR = os.getenv("VIDEO_DIR", "./data/videos")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # e.g. @channelusername or -100123456...
ALERT_SOUND_PATH = os.getenv("ALERT_SOUND_PATH", "alert.wav")  # wav file for local notification

# Detector tuning (MVP)
MIN_CONTOUR_AREA = int(os.getenv("MIN_CONTOUR_AREA", "400"))  # minimal area to consider as "bird"
DETECTION_FRAMES_REQUIRED = int(os.getenv("DETECTION_FRAMES_REQUIRED", "3"))
CLIP_SECONDS = int(os.getenv("CLIP_SECONDS", "6"))
FPS = int(os.getenv("FPS", "15"))

# make dirs
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# ---------- Utilities ----------
def timestamp_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def save_image(frame, prefix="bird"):
    fname = f"{prefix}_{timestamp_str()}.jpg"
    path = os.path.join(IMAGE_DIR, fname)
    cv2.imwrite(path, frame)
    return path

def save_video(frames, fps=FPS, prefix="clip"):
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    fname = f"{prefix}_{timestamp_str()}.mp4"
    path = os.path.join(VIDEO_DIR, fname)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return path

# ---------- Telegram notifier ----------
def send_photo_telegram(photo_path, caption=None):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[telegram] token/chat_id not set, skipping send.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": TELEGRAM_CHAT_ID}
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

def send_video_telegram(video_path, caption=None):
    # optionally: send video to telegram (bot API sendVideo)
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
            with open(video_path, "rb") as vf:
                files = {"video": vf}
                data = {"chat_id": TELEGRAM_CHAT_ID, "caption": "Клип с кормушки"}
                resp = requests.post(url, data=data, files=files, timeout=30)
                resp.raise_for_status()
                print("[telegram] clip sent")
        except Exception as e:
            print("[telegram] failed to send clip:", e)

# ---------- Sound notifier ----------
def play_sound_async(wav_path):
    try:
        wave_obj = sa.WaveObject.from_wave_file(wav_path)
        play_obj = wave_obj.play()  # non-blocking
        # optional: return play_obj if you want to stop later
        return True
    except Exception as e:
        print("[sound] failed to play:", e)
        return False

# ---------- Camera capture thread ----------
class CameraCapture(threading.Thread):
    def __init__(self, src=CAMERA_SOURCE):
        super().__init__(daemon=True)
        self.src = src
        self.cap = None
        self.running = False
        self.frame = None
        self.lock = threading.Lock()

    def run(self):
        print("[camera] starting capture from", self.src)
        self.cap = cv2.VideoCapture(self.src)
        # set FPS if possible
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
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
            time.sleep(1.0 / FPS)
        self.cap.release()
        print("[camera] stopped")

    def get_frame(self):
        with self.lock:
            frame = self.frame  # just reference, no copy
        if frame is None:
            return None
        print("[camera] get_frame copy")
        # copy outside lock
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

    def __init__(self, camera: CameraCapture):
        super().__init__(daemon=True)
        self.camera = camera
        # self.bg = cv2.createBackgroundSubtractorMOG2(
        #     history=200, varThreshold=25, detectShadows=True
        # )

        self.bg = cv2.createBackgroundSubtractorKNN(
            history=200,
            dist2Threshold=1000,
            detectShadows=False
        )

        self.state = self.STATE_IDLE
        self.trigger_counter = 0

        self.buffer = collections.deque(maxlen=CLIP_SECONDS * FPS)
        self.lock = threading.Lock()

        # настройки
        self.TRIGGER_FRAMES = DETECTION_FRAMES_REQUIRED
        self.COOLDOWN_TIME = 3  # секунды между событиями

    # --------------------------
    # ДЕТЕКТОР (можно заменить на ML)
    # --------------------------
    def _detect_movement(self, frame):
        print("|", end="")
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        print("|", end="")
        mask = self.bg.apply(small)
        print("|", end="")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        print("|", end="")
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        print("|", end="")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("|", end="")
        for c in contours:
            if cv2.contourArea(c) >= MIN_CONTOUR_AREA:
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
                # накапливаем буфер
                self.buffer.append(frame)
                print("[detector] buffer appended", end="")
                movement = self._detect_movement(frame)
                print()
                print(f"[detector] state={self.state} movement={movement} counter={self.trigger_counter}")

                # --------------------------
                # СОСТОЯНИЕ: IDLE
                # --------------------------
                if self.state == self.STATE_IDLE:
                    if movement:
                        self.trigger_counter += 1
                    else:
                        self.trigger_counter = max(0, self.trigger_counter - 1)

                    if self.trigger_counter >= self.TRIGGER_FRAMES:
                        # птица появилась!
                        self._trigger_event(frame)
                        print("[detector] TRIGGERED!")
                        self.state = self.STATE_TRIGGERED
                        self.trigger_counter = 0
                    continue

                # --------------------------
                # СОСТОЯНИЕ: TRIGGERED
                # --------------------------
                if self.state == self.STATE_TRIGGERED:
                    # в этом состоянии мы ждём завершения записи клипа
                    # запись клипа работает в отдельном потоке
                    # мы сразу переходим в COOLDOWN
                    print("[detector] cooldown started")
                    self.state = self.STATE_COOLDOWN
                    self.cooldown_start = time.time()
                    continue

                # --------------------------
                # СОСТОЯНИЕ: COOLDOWN
                # --------------------------
                if self.state == self.STATE_COOLDOWN:
                    if time.time() - self.cooldown_start >= self.COOLDOWN_TIME:
                        print("[detector] cooldown ended")
                        # готово к следующей птице
                        self.state = self.STATE_IDLE
                        self.trigger_counter = 0
                    continue

                time.sleep(1.0 / FPS)
        except Exception as e:
            print("[detector] ERROR:", e)
            import traceback
            traceback.print_exc()
            # чтобы поток не умирал
            time.sleep(0.5)

    # --------------------------
    # СОБЫТИЕ ТРИГГЕРА
    # --------------------------
    def _trigger_event(self, frame):
        print("[detector] BIRD EVENT TRIGGERED")

        # сохраняем фото
        image_path = save_image(frame, prefix="bird")

        # отправка фото в телеграм — отдельный поток
        threading.Thread(
            target=send_photo_telegram,
            args=(image_path, "Птица у кормушки!"),
            daemon=True
        ).start()

        # FIXME: после воспроизведения звука все повисает
        # звук
        # threading.Thread(
        #     target=play_sound_async,
        #     args=(ALERT_SOUND_PATH,),
        #     daemon=True
        # ).start()

        # запись клипа — отдельный поток
        threading.Thread(
            target=self._write_clip,
            daemon=True
        ).start()

    # --------------------------
    # ЗАПИСЬ КЛИПА
    # --------------------------
    def _write_clip(self):
        print("[detector] recording clip...")

        with self.lock:
            frames = list(self.buffer)
        # FIXME: второе видео имеет 6 секунд паузы в начале
        # добавить будущие кадры
        extra = CLIP_SECONDS * FPS
        for _ in range(extra):
            f = self.camera.get_frame()
            if f is not None:
                frames.append(f)
            time.sleep(1.0 / FPS)

        path = save_video(frames, fps=FPS, prefix="birdclip")
        print("[detector] clip saved:", path)

        # FIXME: не отправлчется
        # можно отправить в телеграм (опционально)
        threading.Thread(
            target=send_video_telegram,
            args=(path,),
            daemon=True
        ).start()

        # очищаем буфер после записи
        with self.lock:
            self.buffer.clear()

# ---------- Main ----------
def main():
    cam = CameraCapture(src=CAMERA_SOURCE)
    cam.start()
    detector = Detector(camera=cam)
    detector.start()
    print("MVP running. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        cam.stop()
        time.sleep(0.5)

if __name__ == "__main__":
    main()
