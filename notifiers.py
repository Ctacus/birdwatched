"""
Notification utilities: Telegram and sound alerts.
"""

import logging
from typing import Optional

import requests
# import simpleaudio as sa

# import asyncio
# from play_sounds import play_file_async
from playsound3 import playsound

from config import AppConfig

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def send_photo(self, photo_path: str, caption: Optional[str] = None) -> bool:
        if not self.cfg.telegram_bot_token or not self.cfg.telegram_chat_id:
            logger.warning("Token/chat_id not set, skipping send.")
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
                logger.info(f"Photo sent: {photo_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to send photo: {e}", exc_info=True)
                return False

    def send_video(self, video_path: str, caption: Optional[str] = None) -> bool:
        if not self.cfg.telegram_bot_token or not self.cfg.telegram_chat_id:
            logger.warning("Token/chat_id not set, skipping send.")
            return False
        try:
            url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendVideo"
            with open(video_path, "rb") as vf:
                files = {"video": vf}
                data = {"chat_id": self.cfg.telegram_chat_id, "caption": caption or "Птичка!"}
                resp = requests.post(url, data=data, files=files, timeout=120)
                resp.raise_for_status()
                logger.info("Clip sent")
                return True
        except Exception as e:
            logger.error(f"Failed to send clip: {e}", exc_info=True)
            return False


class SoundNotifier:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg


    def playsound(self) -> bool:
        try:
            playsound(sound=self.cfg.alert_sound_path, block=False)
            return True
        except Exception as e:
            logger.error(f"Failed to play sound: {e}", exc_info=True)
            return False