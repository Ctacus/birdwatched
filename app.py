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
from rtsp_camera import RTSPCameraCapture

logger = logging.getLogger(__name__)


class BirdWatcherApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.storage = StorageManager(cfg)
        self.telegram = TelegramNotifier(cfg)
        self.sound = SoundNotifier(cfg)
        # self.camera = CameraCapture(cfg)

        self.camera  =  RTSPCameraCapture(
            cfg,
            rtsp_url="rtsp://192.168.1.78:8080/h264.sdp",
            reconnect_delay=3.0,
            max_reconnect_attempts=-1,  # Infinite retries
            use_ffmpeg_backend=False,    # Better RTSP support
            rtsp_transport="tcp"        # or "udp" for lower latency (less reliable)
        )

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

