import subprocess
import time
import cv2
import numpy as np


class TelegramStaticImageStreamer:
    def __init__(
        self,
        image_path: str,
        rtmp_url: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        bitrate: str = "2000k",
    ):
        self.image_path = image_path
        self.rtmp_url = rtmp_url
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.proc: subprocess.Popen | None = None

    def _ffmpeg_cmd(self):
        return [
            "ffmpeg",
            "-loglevel", "info",

            # INPUT: raw frames from stdin
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",

            # ENCODING
            "-c:v", "libx264",
            "-profile:v", "baseline",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-b:v", self.bitrate,
            "-maxrate", self.bitrate,
            "-bufsize", "4000k",
            "-g", str(self.fps * 2),
            "-sc_threshold", "0",

            # OUTPUT
            "-f", "flv",
            self.rtmp_url,
        ]

    def start(self):
        # Load image
        img = cv2.imread(self.image_path)
        if img is None:
            raise RuntimeError("Failed to load image")

        img = cv2.resize(img, (self.width, self.height))
        img = np.ascontiguousarray(img, dtype=np.uint8)
        frame_bytes = img.tobytes()

        print("Starting FFmpeg…")
        self.proc = subprocess.Popen(
            self._ffmpeg_cmd(),
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        frame_interval = 1.0 / self.fps
        next_time = time.monotonic()

        try:
            while True:
                self.proc.stdin.write(frame_bytes)

                # next_time += frame_interval
                # sleep_time = next_time - time.monotonic()
                # if sleep_time > 0:
                #     time.sleep(sleep_time/2)

                if self.proc.poll() is not None:
                    print("FFmpeg exited")
                    break

        except KeyboardInterrupt:
            print("Stopping…")
        finally:
            self.stop()

    def stop(self):
        if not self.proc:
            return
        try:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


if __name__ == "__main__":
    streamer = TelegramStaticImageStreamer(
        image_path="bird.jpg",
        rtmp_url="rtmps://dc4-1.rtmp.t.me/s/3234968082:IrMYQ3O_BVrXDH_2G5Djag",
    )
    streamer.start()