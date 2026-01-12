"""
birdwatcher_mvp.py
MVP: монитор кормушки — детекция (MOG2/KNN), сохранение фото/видео,
отправка фото в Telegram, воспроизведение звука.
Настройки через environment variables или .env (см. пример ниже).
"""

import os

from app import BirdWatcherApp
from config import AppConfig, setup_logging


def main():
    cfg = AppConfig.from_env()

    # Setup logging with level from environment or default to INFO
    log_level = os.getenv("LOG_LEVEL", "INFO")
    setup_logging(log_level)

    app = BirdWatcherApp(cfg)
    app.start()


if __name__ == "__main__":
    main()
