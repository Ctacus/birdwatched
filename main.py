"""
birdwatcher_mvp.py
MVP: монитор кормушки — детекция (MOG2/KNN), сохранение фото/видео,
отправка фото в Telegram, воспроизведение звука.
Настройки через environment variables или .env (см. пример ниже).
"""

from app import BirdWatcherApp
from config import AppConfig


def main():
    cfg = AppConfig.from_env()
    app = BirdWatcherApp(cfg)
    app.start()


if __name__ == "__main__":
    main()
