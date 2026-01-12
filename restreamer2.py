"""
Telegram RTMP restreamer.

Streams raw frames from BaseCameraCapture to Telegram RTMPS
using FFmpeg and software x264 encoding.
"""

import logging
import subprocess
import threading
import time
from typing import Optional

import cv2
import numpy as np

from base_camera import BaseCameraCapture
from config import AppConfig

logger = logging.getLogger(__name__)


class TelegramRTMPRestreamer2(threading.Thread):
    def __init__(
        self,
        cfg: AppConfig,
        camera_source: BaseCameraCapture,
        rtmp_server_url: Optional[str] = None,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = -1,
        video_width: Optional[int] = None,
        video_height: Optional[int] = None,
        bitrate: str = "2000k",
        preset: str = "veryfast",
    ):
        super().__init__(daemon=True)

        if not cfg.telegram_rtmp_stream_key:
            raise ValueError("telegram_rtmp_stream_key is not set")

        self.cfg = cfg
        self.camera = camera_source

        server = rtmp_server_url or cfg.telegram_rtmp_server_url
        if not server.endswith("/"):
            server += "/"
        self.rtmp_url = server + cfg.telegram_rtmp_stream_key

        self.video_width = video_width
        self.video_height = video_height
        self.bitrate = bitrate
        self.preset = preset

        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_attempts = 0

        self.ffmpeg: Optional[subprocess.Popen] = None
        self.stderr_thread: Optional[threading.Thread] = None

        self.running = False
        self.streaming = False

    # ------------------------------------------------------------------ #
    # FFmpeg
    # ------------------------------------------------------------------ #

    def _ffmpeg_cmd(self, width: int, height: int, fps: int) -> list[str]:
        return [
            "ffmpeg",
            "-loglevel", "info",

            # INPUT
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-use_wallclock_as_timestamps", "1",
            "-i", "-",

            # ENCODING
            "-c:v", "libx264",
            "-profile:v", "baseline",
            "-preset", self.preset,
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-b:v", self.bitrate,
            "-maxrate", self.bitrate,
            "-bufsize", f"{int(self.bitrate[:-1]) * 2}k",
            "-g", str(fps * 2),
            "-keyint_min", str(fps * 2),
            "-sc_threshold", "0",

            # OUTPUT
            "-f", "flv",
            self.rtmp_url,
        ]

    def _start_ffmpeg(self, width: int, height: int, fps: int) -> bool:
        cmd = self._ffmpeg_cmd(width, height, fps)
        logger.info("Starting FFmpeg → Telegram RTMP")

        self.ffmpeg = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        self.stderr_thread = threading.Thread(
            target=self._read_ffmpeg_stderr,
            daemon=True,
        )
        self.stderr_thread.start()

        time.sleep(1.0)

        if self.ffmpeg.poll() is not None:
            logger.error("FFmpeg failed to start")
            return False

        self.streaming = True
        return True

    def _stop_ffmpeg(self):
        if not self.ffmpeg:
            return

        logger.info("Stopping FFmpeg")
        try:
            if self.ffmpeg.stdin:
                self.ffmpeg.stdin.close()
            self.ffmpeg.terminate()
            self.ffmpeg.wait(timeout=2)
        except Exception:
            self.ffmpeg.kill()
        finally:
            self.ffmpeg = None
            self.streaming = False

    def _read_ffmpeg_stderr(self):
        if not self.ffmpeg or not self.ffmpeg.stderr:
            return

        for line in iter(self.ffmpeg.stderr.readline, b""):
            msg = line.decode("utf-8", errors="ignore").strip()
            if not msg:
                continue

            if any(x in msg.lower() for x in ("error", "failed", "invalid")):
                logger.error("[FFmpeg] %s", msg)
            else:
                logger.info("[FFmpeg] %s", msg)

    # ------------------------------------------------------------------ #
    # Thread main loop
    # ------------------------------------------------------------------ #

    def run(self):
        logger.info("Telegram RTMP restreamer started")
        self.running = True

        # Grab first valid frame
        frame = None
        while self.running and frame is None:
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.5)

        if frame is None:
            logger.error("No frames available from camera")
            return

        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        src_h, src_w = frame.shape[:2]

        width = self.video_width or src_w
        height = self.video_height or src_h
        fps = self.cfg.fps
        logger.info("_start_ffmpeg")
        if not self._start_ffmpeg(width, height, fps):
            return


        logger.info("running...")
        while self.running:
            frame = self.camera.get_frame()
            if frame is None:
                continue

            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))

            frame = np.ascontiguousarray(frame, dtype=np.uint8)

            try:
                self.ffmpeg.stdin.write(frame.tobytes())
            except (BrokenPipeError, AttributeError):
                logger.warning("FFmpeg pipe broken")
            if not self._reconnect(width, height, fps):
                break



        self._stop_ffmpeg()
        logger.info("Telegram RTMP restreamer stopped")

    # ------------------------------------------------------------------ #
    # Reconnect logic
    # ------------------------------------------------------------------ #

    def _reconnect(self, width: int, height: int, fps: int) -> bool:
        if self.max_reconnect_attempts > 0:
            if self.reconnect_attempts >= self.max_reconnect_attempts:
                logger.error("Max reconnect attempts reached")
                return False

        self.reconnect_attempts += 1
        logger.info("Reconnecting to Telegram RTMP (%d)", self.reconnect_attempts)

        self._stop_ffmpeg()
        time.sleep(self.reconnect_delay)
        return self._start_ffmpeg(width, height, fps)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def stop(self):
        self.running = False

    def is_streaming(self) -> bool:
        return (
            self.streaming
            and self.ffmpeg is not None
            and self.ffmpeg.poll() is None
        )
