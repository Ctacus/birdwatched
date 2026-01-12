"""
Notification utilities: Telegram and sound alerts.
"""

from typing import Optional

import requests
import simpleaudio as sa

from config import AppConfig


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

