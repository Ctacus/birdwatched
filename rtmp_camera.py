"""
RTMP stream camera capture thread with reconnection support.
"""

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from config import AppConfig

logger = logging.getLogger(__name__)


class RTMPCameraCapture(threading.Thread):
    """
    Camera capture from RTMP video stream with automatic reconnection.
    
    This class is designed specifically for RTMP streams and includes:
    - Automatic reconnection on connection loss
    - Configurable retry delays
    - Better error handling for network issues
    - Buffer management for stream stability
    """
    
    def __init__(self, cfg: AppConfig, rtmp_url: Optional[str] = None, reconnect_delay: float = 5.0, max_reconnect_attempts: int = -1):
        """
        Initialize RTMP camera capture.
        
        Args:
            cfg: Application configuration
            rtmp_url: RTMP stream URL (if None, uses cfg.camera_source)
            reconnect_delay: Seconds to wait before reconnecting (default: 5.0)
            max_reconnect_attempts: Maximum reconnection attempts (-1 for infinite, default: -1)
        """
        super().__init__(daemon=True)
        self.cfg = cfg
        self.rtmp_url = rtmp_url or (cfg.camera_source if isinstance(cfg.camera_source, str) else None)
        if not self.rtmp_url:
            raise ValueError("RTMP URL must be provided either as rtmp_url parameter or in cfg.camera_source")
        
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_attempts = 0
        
        self.cap = None
        self.running = False
        self.frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.connected = False
        
    def _connect(self) -> bool:
        """
        Attempt to connect to the RTMP stream.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to {self.rtmp_url}...")
            self.cap = cv2.VideoCapture(self.rtmp_url)
            
            # Set buffer size to reduce latency (optional, may help with some streams)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set FPS if configured
            if self.cfg.fps:
                self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
            
            # Try to read a frame to verify connection
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.connected = True
                self.reconnect_attempts = 0
                logger.info("Connected successfully")
                return True
            else:
                logger.warning("Connection failed: unable to read frame")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}", exc_info=True)
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
            return False
    
    def _reconnect(self) -> bool:
        """
        Attempt to reconnect to the RTMP stream.
        
        Returns:
            True if reconnection successful, False otherwise
        """
        if self.max_reconnect_attempts > 0 and self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            return False
        
        self.reconnect_attempts += 1
        logger.info(f"Reconnection attempt {self.reconnect_attempts}...")
        
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        
        self.connected = False
        time.sleep(self.reconnect_delay)
        return self._connect()
    
    def run(self):
        """Main capture loop with reconnection logic."""
        logger.info(f"Starting RTMP capture from {self.rtmp_url}")
        
        # Initial connection
        if not self._connect():
            logger.error("Initial connection failed")
            return
        
        self.running = True
        consecutive_failures = 0
        max_consecutive_failures = 10  # Number of consecutive read failures before attempting reconnect
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"{consecutive_failures} consecutive read failures, attempting reconnect...")
                        if not self._reconnect():
                            if self.max_reconnect_attempts > 0 and self.reconnect_attempts >= self.max_reconnect_attempts:
                                logger.error("Stopping due to max reconnection attempts")
                                break
                            time.sleep(self.reconnect_delay)
                            continue
                        consecutive_failures = 0
                    else:
                        time.sleep(0.1)
                    continue
                
                # Successful frame read
                consecutive_failures = 0
                with self.lock:
                    self.frame = frame.copy()
                
                # Control frame rate
                time.sleep(1.0 / self.cfg.fps)
                
            except Exception as e:
                logger.error(f"Error during capture: {e}", exc_info=True)
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    if not self._reconnect():
                        if self.max_reconnect_attempts > 0 and self.reconnect_attempts >= self.max_reconnect_attempts:
                            logger.error("Stopping due to max reconnection attempts")
                            break
                        time.sleep(self.reconnect_delay)
                else:
                    time.sleep(0.1)
        
        # Cleanup
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
        logger.info("RTMP camera capture stopped")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest captured frame.
        
        Returns:
            Latest frame as numpy array, or None if no frame available
        """
        with self.lock:
            frame = self.frame
        if frame is None:
            return None
        return np.copy(frame)
    
    def stop(self):
        """Stop the capture thread."""
        self.running = False
    
    def is_connected(self) -> bool:
        """Check if currently connected to the stream."""
        return self.connected

