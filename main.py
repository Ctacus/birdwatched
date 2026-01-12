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
            ret, frame = self.cap.read()
            if not ret:
                print("[camera] frame not read, sleeping briefly...")
                time.sleep(0.1)
                continue
            with self.lock:
                self.frame = frame.copy()
            time.sleep(1.0 / FPS)
        self.cap.release()
        print("[camera] stopped")

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False

# ---------- Detector thread ----------
class Detector(threading.Thread):
    def __init__(self, camera: CameraCapture):
        super().__init__(daemon=True)
        self.camera = camera
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=True)
        self.detected_frames = 0
        self.buffer = collections.deque(maxlen=CLIP_SECONDS * FPS)
        self.recording = False
        self.lock = threading.Lock()

    def run(self):
        print("[detector] started")
        while True:
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            # small preview for processing
            proc = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            fgmask = self.bg_subtractor.apply(proc)
            # morphological ops
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
            # find contours
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            found = False
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= MIN_CONTOUR_AREA:
                    found = True
                    break

            # maintain buffer of last frames for clip
            self.buffer.append(frame)

            if found:
                self.detected_frames += 1
            else:
                self.detected_frames = max(0, self.detected_frames - 1)

            if self.detected_frames >= DETECTION_FRAMES_REQUIRED and not self.recording:
                # trigger start of event
                print("[detector] bird detected! creating snapshot/clip.")
                self.recording = True
                # save immediate snapshot
                image_path = save_image(frame, prefix="bird")
                # send to telegram in background
                threading.Thread(target=send_photo_telegram, args=(image_path, "Птица у кормушки!"), daemon=True).start()
                # play alert sound locally
                threading.Thread(target=play_sound_async, args=(ALERT_SOUND_PATH,), daemon=True).start()

                # start clip writer in background: gather current buffer + next CLIP_SECONDS frames
                threading.Thread(target=self._write_clip_from_buffer, daemon=True).start()

            # small sleep
            time.sleep(1.0 / FPS)

    def _write_clip_from_buffer(self):
        # copy buffer content
        with self.lock:
            frames_to_save = list(self.buffer)
        # append next CLIP_SECONDS*FPS frames
        frames_needed = CLIP_SECONDS * FPS
        count = 0
        while count < frames_needed:
            f = self.camera.get_frame()
            if f is not None:
                frames_to_save.append(f)
                count += 1
            time.sleep(1.0 / FPS)
        video_path = save_video(frames_to_save, fps=FPS, prefix="birdclip")
        print("[detector] saved clip:", video_path)
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
        # reset
        self.recording = False
        # clear buffer to avoid duplicates
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
