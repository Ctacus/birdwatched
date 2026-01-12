"""
Application wiring for the bird watcher MVP.
"""
import asyncio
import logging
import os
import time

from camera import CameraCapture
from config import AppConfig, setup_logging
from detector import Detector
from notifiers import SoundNotifier, TelegramNotifier
from plain_restreamer import FFmpegStreamer
from restreamer2 import TelegramRTMPRestreamer2
from storage import StorageManager
from rtsp_camera import RTSPCameraCapture
from telegram_rtmp_restreamer import TelegramRTMPRestreamer

logger = logging.getLogger(__name__)


class BirdWatcherApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.storage = StorageManager(cfg)
        self.telegram = TelegramNotifier(cfg)
        self.sound = SoundNotifier(cfg)
        self.running = True
        self.controller = None
        # self.camera = CameraCapture(cfg)

        self.camera  =  RTSPCameraCapture(
            cfg,
            rtsp_url=cfg.rtsp_url,
            reconnect_delay=3.0,
            max_reconnect_attempts=-1,  # Infinite retries
            use_ffmpeg_backend=False,    # Better RTSP support
            rtsp_transport="tcp"        # or "udp" for lower latency (less reliable)
        )

        # self.restreamer = TelegramRTMPRestreamer(
        #     cfg=cfg,
        #     camera_source=self.camera,
        #     bitrate="2000k",
        #     preset="veryfast"
        # )

        self.restreamer = FFmpegStreamer(
            rtsp_url=cfg.rtsp_url,
            rtmps_url= f"{cfg.telegram_rtmp_server_url}{cfg.telegram_rtmp_stream_key}"
        )

        self.detector = Detector(cfg, self.camera, self.storage, self.telegram, self.sound)

    def start(self):

        # asyncio.run(self.controller.start())
        self.camera.start()
        if self.cfg.enable_stream:
            self.restreamer.start()
        if self.cfg.enable_detector:
            self.detector.start()

        # import bot_controller
        # self.controller = bot_controller.BotController(self.cfg, self)
        logger.info("MVP running. Ctrl+C to stop.")
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
        logger.info("App stopped.")

    def stop(self):
        self.running = False
        logger.info("Stopping...")
        self.camera.stop()
        if self.cfg.enable_stream:
            self.restreamer.stop()
        time.sleep(0.5)

