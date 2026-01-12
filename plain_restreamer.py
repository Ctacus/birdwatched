import subprocess
import threading
import logging
from typing import Optional


class FFmpegStreamer:
    def __init__(
        self,
        rtsp_url: str,
        rtmps_url: str,
        crf: int = 38,

        preset: str = "veryfast",
        ffmpeg_path: str = "ffmpeg",
    ):
        self.rtsp_url = rtsp_url
        self.rtmps_url = rtmps_url
        self.crf = crf
        self.preset = preset
        self.ffmpeg_path = ffmpeg_path

        self.process: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None

        self.logger = logging.getLogger(self.__class__.__name__)

    def _build_command(self) -> list[str]:
        return [
            "wsl", self.ffmpeg_path,
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-c:v", "libx264",
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-f", "flv",
            self.rtmps_url,
        ]

    def start(self) -> None:
        if self.is_running():
            self.logger.info("FFmpeg already running")
            return

        cmd = self._build_command()
        self.logger.info("Starting FFmpeg: %s", " ".join(cmd))

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            daemon=True,
        )
        self._stderr_thread.start()

    def _read_stderr(self) -> None:
        assert self.process and self.process.stderr

        for line in self.process.stderr:
            pass
            # self.logger.debug("ffmpeg: %s", line.rstrip())

    def stop(self) -> None:
        if not self.process:
            return

        self.logger.info("Stopping FFmpeg")
        self.process.terminate()
        self.process.wait(timeout=5)

        self.process = None
        self._stderr_thread = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None
