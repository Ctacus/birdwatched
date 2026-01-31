"""
Application wiring for the bird watcher MVP.
"""
import asyncio
import logging
import os
import time
import threading

import cv2
import numpy as np

from camera import CameraCapture
from config import AppConfig, setup_logging
from detector import Detector
from notifiers import SoundNotifier, TelegramNotifier
from plain_restreamer import FFmpegStreamer
from restreamer2 import TelegramRTMPRestreamer2
from storage import StorageManager
from rtsp_camera import RTSPCameraCapture
from telegram_rtmp_restreamer import TelegramRTMPRestreamer
from frame_filters import FilterChain, WeatherTextOverlayFilter
from weather_service import WeatherService, WeatherScheduler

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

        # Get weather configuration from environment or use defaults
        weather_latitude = float(os.getenv("WEATHER_LATITUDE", "53.199821"))  # Default: Samara, Russia
        weather_longitude = float(os.getenv("WEATHER_LONGITUDE", "50.1302682"))
        enable_weather = bool(int(os.getenv("ENABLE_WEATHER", "1")))
        
        # Set up filter chain
        filter_chain = FilterChain()

        if enable_weather:
            # Initialize weather service (Open-Meteo API, no API key required)
            weather_service = WeatherService(
                latitude=weather_latitude,
                longitude=weather_longitude,
                units="metric",
                update_interval=300.0,  # 5 minutes
            )
            weather_scheduler = WeatherScheduler(
                weather_service=weather_service,
                update_interval=300.0,  # 5 minutes
            )
            
            # Create weather overlay filter (auto-positions to top right)
            weather_filter = WeatherTextOverlayFilter(
                weather_service=weather_service,
                position="top_right",  # Auto-position to top right corner
                font_scale=0.8,
                color=(255, 235, 155),  # White text
                thickness=2,
                transparency=0.8,
                background=False,
                background_color=(0, 0, 0),  # Black background
                background_transparency=0.7,
            )
            filter_chain.add_filter(weather_filter)
            
            # Store references for scheduler management
            self.weather_scheduler = weather_scheduler
        else:
            self.weather_scheduler = None
        
        self.camera = RTSPCameraCapture(
            cfg,
            rtsp_url=cfg.rtsp_url,
            reconnect_delay=3.0,
            max_reconnect_attempts=-1,  # Infinite retries
            use_ffmpeg_backend=False,    # Better RTSP support
            rtsp_transport="tcp",        # or "udp" for lower latency (less reliable)
            filter_chain=filter_chain
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
        
        # Start weather scheduler if enabled
        if self.weather_scheduler:
            self.weather_scheduler.start()
        
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
        
        # Stop weather scheduler if enabled
        if self.weather_scheduler:
            self.weather_scheduler.stop()
        
        self.camera.stop()
        if self.cfg.enable_stream:
            self.restreamer.stop()
        time.sleep(0.5)

