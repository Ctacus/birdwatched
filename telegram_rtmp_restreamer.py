"""
RTMP restreamer to Telegram channel via RTMP.
Minimal MVP for testing.
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


class TelegramRTMPRestreamer(threading.Thread):
    """Restreams video from a camera source to Telegram channel via RTMP."""
    
    def __init__(
        self,
        cfg: AppConfig,
        camera_source: BaseCameraCapture,
        bitrate: str = "2000k",
        preset: str = "veryfast",
    ):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.camera_source = camera_source
        self.rtmp_url = f"{cfg.telegram_rtmp_server_url}{cfg.telegram_rtmp_stream_key}"
        self.bitrate = bitrate
        self.preset = preset
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.running = False
        
    def _build_ffmpeg_command(self, width: int, height: int, fps: int) -> list:
        return [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", self.preset,
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-b:v", self.bitrate,
            "-g", str(fps),              # Keyframe every 1 second (faster start)
            "-keyint_min", str(fps),     # Min keyframe interval
            "-force_key_frames", f"expr:gte(t,n_forced*1)",  # Force keyframe every 1s
            "-f", "flv",
            "-flvflags", "no_duration_filesize",
            self.rtmp_url,
        ]
    
    def run(self):
        """Main streaming loop."""
        logger.info(f"Starting stream to {self.rtmp_url}")
        
        # Discard stale frames - consume what's cached
        for _ in range(10):
            f = self.camera_source.get_frame()
            if f is not None:
                break
            time.sleep(0.05)
        
        # Get fresh frame
        frame = None
        while frame is None:
            frame = self.camera_source.get_frame()
            if frame is None:
                time.sleep(0.05)
        
        height, width = frame.shape[:2]
        fps = self.cfg.fps
        
        logger.info(f"Resolution: {width}x{height}, FPS: {fps}")
        
        # Start FFmpeg
        cmd = self._build_ffmpeg_command(width, height, fps)
        logger.info(f"FFmpeg command: {' '.join(cmd)}")
        
        self.ffmpeg_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Start stderr reader thread
        def read_stderr():
            for line in self.ffmpeg_process.stderr:
                logger.info(f"[FFmpeg] {line.decode('utf-8', errors='ignore').strip()}")
        
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()
        
        self.running = True
        frame_interval = 1.0 / fps
        next_frame_time = time.monotonic()
        last_frame = frame  # Keep last valid frame
        
        while self.running:
            # Wait until it's time for the next frame
            now = time.monotonic()
            sleep_time = next_frame_time - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Get latest frame from camera
            new_frame = self.camera_source.get_frame()
            if new_frame is not None:
                # Resize if needed
                if new_frame.shape[:2] != (height, width):
                    new_frame = cv2.resize(new_frame, (width, height))
                last_frame = new_frame
            
            # Always write a frame at the expected interval (repeat last if no new)
            try:
                self.ffmpeg_process.stdin.write(last_frame.tobytes())
            except BrokenPipeError:
                logger.error("FFmpeg pipe broken")
                break
            
            next_frame_time += frame_interval
        
        # Cleanup
        if self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
        logger.info("Restreamer stopped")
    
    def stop(self):
        """Stop the streaming thread."""
        self.running = False


