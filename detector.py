"""
Movement detector with simple state machine.
"""

import collections
import logging
import threading
import time
from typing import Deque, Optional
from xmlrpc.client import Error

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from base_camera import BaseCameraCapture
from config import AppConfig
from notifiers import SoundNotifier, TelegramNotifier
from storage import StorageManager

logger = logging.getLogger(__name__)

def draw_plot(
        values,
        width=400,
        height=200,
        ymin=-0.2,
        ymax=1.2,
        color=(0, 255, 0)
):
    img = np.zeros((height, width, 3), dtype=np.uint8)

    if len(values) < 2:
        return img

    vals = np.array(values, dtype=np.float32)

    # normalize Y
    vals = np.clip(vals, ymin, ymax)
    vals = (vals - ymin) / (ymax - ymin)
    vals = height - (vals * height)

    # X coordinates
    xs = np.linspace(0, width - 1, len(vals)).astype(np.int32)
    pts = np.column_stack((xs, vals.astype(np.int32)))

    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=2)
    return img


class ClipBuffer:
    def __init__(self, fps: int, clip_seconds: int):
        self.fps = fps  # Store fps for the weighting logic

        self.max_clip_length = clip_seconds * fps
        self.buffer: Deque[np.ndarray] = collections.deque(maxlen=self.max_clip_length + 120)
        self.motion_flags: Deque[float] = collections.deque(maxlen=self.buffer.maxlen)
        self.motion_percent_log: Deque[float] = collections.deque(maxlen=self.buffer.maxlen)
        if self.buffer.maxlen <= self.max_clip_length:
            raise Error("Buffer must be at least 1 frame wider than clip")
        self.window_totals: Deque[float] = collections.deque(maxlen=self.buffer.maxlen - self.max_clip_length + 1)
        self.window_totals.append(0) # initial window total
        self._average_frame: Optional[np.ndarray] = None
        self.global_frame_count: int = 0



    def debug_setup_plot(self):
        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)

        ax.set_xlim(0, self.buffer.maxlen)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")

        def update(frame):
            data = self.motion_flags
            # X axis = elapsed time
            x = np.linspace(
                0,
                len(data)
            )

            line.set_data(x, list(data))
            ax.set_xlim(x[0], x[-1])

            return line,


        ani = FuncAnimation(
            fig,
            update,
            interval=33,  # ms
            blit=True
        )

        plt.show()

    def debug_output(self):
        cv2.imshow("Live Average Frame", self.average_frame)
        cv2.imshow("Current Frame", self.buffer[-1])
        cv2.imshow("Activation", draw_plot(self.motion_flags, width=len(self.motion_flags) * 2))
        cv2.imshow("Window Totals", draw_plot(self.window_totals, ymax=300, width=len(self.window_totals) * 2))

        self.motion_percent_log.append(self.motion_percent())

        cv2.imshow("Motion percent", draw_plot(self.motion_percent_log, width=len(self.motion_percent_log) * 2))

        cv2.waitKey(1)



    def append(self, frame: np.ndarray, motion: float):
        if motion > 1 or motion < 0:
            logger.warning(f"Incorrect motion value: {motion}! (expected in range [0;1])")
            motion = 0
        self.buffer.append(frame)
        self.global_frame_count += 1
        self.motion_flags.append(motion)
        if len(self.buffer) <= self.max_clip_length: # initial windows accumulation
            self.window_totals[0] += motion
        else:
            dropped_motion_idx = -self.max_clip_length - 1
            total_motion = self.window_totals[-1] + motion - self.motion_flags[dropped_motion_idx]
            self.window_totals.append(total_motion)

        # Formula: New_Avg = (1 - w) * Old_Avg + w * New_Frame
        weight = 1.0 / self.fps / self.fps

        if self._average_frame is None:
            # Initialize with the first frame (converted to float for precision)
            self._average_frame = frame.astype(np.float32)
        else:
            # Update the running average
            # Note: frame is cast to float32 to match the average buffer
            self._average_frame = ((1.0 - weight) * self._average_frame) + (weight * frame.astype(np.float32))

        self.debug_output()


        self.trim_start(2)


    def trim_start(self, frame_cnt: int, threshold: float=0.1):
        for _ in range(frame_cnt):
            if len(self.buffer) == 0:
                return
            motion_level = self.motion_flags[0]
            if motion_level >= threshold:
                return
            self.buffer.popleft()
            self.motion_flags.popleft()
            if len(self.window_totals) > 1:
                self.window_totals.popleft()
            else:
                self.window_totals[0] -= motion_level  # на будущее, пока бессмысленно


    def motion_percent(self) -> float:
        return max(self.window_totals) / self.max_clip_length

    def is_ready(self) -> bool:
        return len(self.buffer) == self.buffer.maxlen

    def get_clip(self):
        best_windows_motion = max(self.window_totals)
        idx = self.window_totals.index(best_windows_motion)
        return list(self.buffer)[idx:idx + self.max_clip_length]

    @property
    def average_frame(self) -> np.ndarray:
        """Returns the current running average frame."""
        return self._average_frame.astype(np.uint8)



class Detector(threading.Thread):
    """
    Детектор с машиной состояний:
    IDLE -> TRIGGERED -> COOLDOWN -> IDLE -> ...
    """

    STATE_IDLE = "idle"
    STATE_TRIGGERED = "triggered"
    STATE_COOLDOWN = "cooldown"

    def __init__(
        self,
        cfg: AppConfig,
        camera: BaseCameraCapture,
        storage: StorageManager,
        telegram: TelegramNotifier,
        sound: SoundNotifier,
    ):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.camera = camera
        self.storage = storage
        self.telegram = telegram
        self.sound = sound

        self.bg = cv2.createBackgroundSubtractorKNN(
            history=200,
            dist2Threshold=1000,
            detectShadows=False,
        )

        self.state = self.STATE_IDLE
        self.trigger_counter = 0
        # self.buffer: Deque[np.ndarray] = collections.deque(maxlen=self.cfg.clip_seconds * self.cfg.fps)
        self.buffer: ClipBuffer = ClipBuffer(self.cfg.fps, self.cfg.clip_seconds)
        self.lock = threading.Lock()
        self.cooldown_start = 0.0
        self.is_recording = False
        # self.trigger_level = cfg.movement_level_required  # стандартный уровень срабатывания
        # self.min_trigger_level = cfg.movement_level_required/2 # минимальный уровень срабатывания
        # self.current_trigger_level = cfg.movement_level_required/2 # минимальный уровень срабатывания

    # --------------------------
    def _detect_movement(self, frame: np.ndarray) -> bool:
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        mask = self.bg.apply(small)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) >= self.cfg.min_contour_area:
                return True
        return False

    # --------------------------
    def run(self):
        logger.info("Detector started")
        last_state = None
        last_movement = False
        last_counter = 0
        last_loop = 0

        try:
            while True:
                time.sleep(max(1.0 / self.cfg.fps - last_loop, 1.0 / self.cfg.fps /2))
                start = time.perf_counter()
                # logger.debug("Fetching frame...")
                frame = self.camera.get_frame()
                last_frame_time = self.camera.get_last_frame_time()
                if frame is None:
                    logger.debug("No frame available")
                    time.sleep(1.0 / self.cfg.fps / 2)
                    continue
                movement = self._detect_movement(frame)
                self.buffer.append(frame, movement)

                # save average frame from buffer to avg/bird.jpg
                if self.buffer.global_frame_count % 40000 == 0:
                    logger.debug("saving average frame...")
                    self.storage.save_image(self.buffer.average_frame, prefix="avg/bird")

                delta = time.perf_counter() - start
                camera_delay = time.time() - last_frame_time
                if self.state !=last_state or movement != last_movement or last_counter != self.trigger_counter:
                    buffer_movement = self.buffer.motion_percent()
                    logger.debug(f"state={self.state} movement={movement} counter={self.trigger_counter} buffer_motion_percent={buffer_movement:0.2f} time = {delta:0.4f} last_loop = {last_loop:0.4f}  camera-detector delay={camera_delay:0.4f}")
                last_state, last_movement, last_counter = self.state, movement, self.trigger_counter
                if self.state == self.STATE_IDLE:
                    self._handle_idle(movement, frame)
                    last_loop = time.perf_counter() - start
                    continue

                if self.state == self.STATE_TRIGGERED:
                    logger.info(f"Cooldown started for {self.cfg.cooldown_seconds} seconds")
                    self.state = self.STATE_COOLDOWN
                    self.cooldown_start = time.time()
                    last_loop = time.perf_counter() - start
                    continue

                if self.state == self.STATE_COOLDOWN:
                    self._handle_cooldown()
                    last_loop = time.perf_counter() - start
                    continue

        except Exception as e:
            logger.error(f"Detector error: {e}", exc_info=True)
            time.sleep(1)

    def _handle_idle(self, movement: bool, frame: np.ndarray):
        if movement:
            self.trigger_counter += 1
        else:
            self.trigger_counter = max(0, self.trigger_counter - 1)

        if self.trigger_counter >= self.cfg.detection_frames_required and self.buffer.is_ready():
            motion_pct = self.buffer.motion_percent()
            if motion_pct >= self.cfg.movement_level_required:
                logging.info(f"Motion detected, level: {motion_pct}")
                self._trigger_event(frame)
                self.state = self.STATE_TRIGGERED
                self.trigger_counter = 0

    def _handle_cooldown(self):
        if time.time() - self.cooldown_start >= self.cfg.cooldown_seconds:
            logger.info("Cooldown ended")
            self.state = self.STATE_IDLE
            self.trigger_counter = 0

    # --------------------------
    def _trigger_event(self, frame: np.ndarray):
        logger.info("BIRD EVENT TRIGGERED")

        image_path = self.storage.save_image(frame, prefix="bird")

        if self.cfg.enable_posts:
            threading.Thread(
                target=self.telegram.send_photo,
                args=(image_path, "У нас гости!"),
                daemon=True,
            ).start()

        threading.Thread(target=self.sound.playsound, daemon=True).start()

        if self.is_recording:
            logger.warning("Clip already recording, skip new clip")
            return

        threading.Thread(target=self._write_clip, daemon=True).start()

    # --------------------------
    def _write_clip(self):
        logger.info("Recording clip...")

        self.is_recording = True
        try:
            logger.info(f"Fetching clip from buffer...")
            frames = self.buffer.get_clip()
            path = self.storage.save_video(frames, prefix="birdclip")
            logger.info(f"Clip saved: {path}")

            if self.cfg.enable_posts:
                threading.Thread(
                    target=self.telegram.send_video,
                    args=(path,),
                    daemon=True,
                ).start()
        finally:
            self.is_recording = False

