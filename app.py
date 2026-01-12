"""
Application wiring for the bird watcher MVP.
"""

import time

from camera import CameraCapture
from config import AppConfig
from detector import Detector
from notifiers import SoundNotifier, TelegramNotifier
from storage import StorageManager


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

