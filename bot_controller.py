import asyncio
import logging

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from app import BirdWatcherApp
from detector import Detector


class BotController:
    def __init__(self, cfg, app: BirdWatcherApp):
        self.cfg = cfg
        self.app = app
        self.telegram_app = Application.builder().token(cfg.telegram_bot_token).build()
        self.telegram_app.add_handler(CommandHandler("start", self.start_handler))
        self.telegram_app.add_handler(CommandHandler("stop", self.stop_handler))
        self.telegram_app.run_polling()


    # async def start(self):
    #     logging.info("Started telegram handlers!")
    #     await self.telegram_app.initialize()
    #     await self.telegram_app.start()
    #     await self.telegram_app.updater.start_polling()

    async def start_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Привет!")

    async def stop_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.app.stop()
        await update.message.reply_text("Пока!")
        await self.telegram_app.stop()
        await self.telegram_app.stop_running()
        await self.telegram_app.shutdown()

