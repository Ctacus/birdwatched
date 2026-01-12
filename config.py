"""
Configuration utilities for the bird watcher application.
"""

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv


def setup_logging(level: str = "INFO"):
    """
    Configure logging with timestamps and levels.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)-9s -  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"log_level: {logging.getLevelName(log_level)}")


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
    movement_level_required: float
    clip_seconds: int
    fps: int
    cooldown_seconds: int = 10
    telegram_rtmp_stream_key: str = ""
    telegram_rtmp_server_url: str = "rtmps://dc4-1.rtmp.t.me/s/"

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
            movement_level_required=float(os.getenv("MOVEMENT_LEVEL_REQUIRED", "0.5")),
            clip_seconds=int(os.getenv("CLIP_SECONDS", "6")),
            cooldown_seconds=int(os.getenv("COOLDOWN_SECONDS", "20")),
            fps=int(os.getenv("FPS", "15")),
            telegram_rtmp_stream_key=os.getenv("TELEGRAM_RTMP_STREAM_KEY", ""),
            telegram_rtmp_server_url=os.getenv("TELEGRAM_RTMP_SERVER_URL", "rtmps://dc4-1.rtmp.t.me/s/"),
        )

    @staticmethod
    def _parse_camera_source(value: str) -> int | str:
        try:
            return int(value)
        except ValueError:
            return value

