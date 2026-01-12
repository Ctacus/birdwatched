"""
Application wiring for the bird watcher MVP.
"""

import logging
import os
import time

from camera import CameraCapture
from config import AppConfig, setup_logging
from detector import Detector
from notifiers import SoundNotifier, TelegramNotifier
from storage import StorageManager

logger = logging.getLogger(__name__)


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
        logger.info("MVP running. Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping...")
            self.camera.stop()
            time.sleep(0.5)

