"""
RTSP stream camera capture thread with reconnection support.
Optimized for RTSP/H.264 streams from IP cameras and Android apps.
"""

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from base_camera import BaseCameraCapture
from config import AppConfig

logger = logging.getLogger(__name__)


class RTSPCameraCapture(BaseCameraCapture):
    """
    Camera capture from RTSP video stream with automatic reconnection.
    
    This class is designed specifically for RTSP streams and includes:
    - Automatic reconnection on connection loss
    - Configurable retry delays
    - Better error handling for network issues
    - Low-latency buffer management optimized for RTSP
    - Support for RTSP authentication if needed
    """
    
    def __init__(
        self, 
        cfg: AppConfig, 
        rtsp_url: Optional[str] = None, 
        reconnect_delay: float = 5.0, 
        max_reconnect_attempts: int = -1,
        use_ffmpeg_backend: bool = True,
        rtsp_transport: str = "tcp",
        filter_chain=None,
    ):
        """
        Initialize RTSP camera capture.
        
        Args:
            cfg: Application configuration
            rtsp_url: RTSP stream URL (if None, uses cfg.camera_source)
            reconnect_delay: Seconds to wait before reconnecting (default: 5.0)
            max_reconnect_attempts: Maximum reconnection attempts (-1 for infinite, default: -1)
            use_ffmpeg_backend: Use FFMPEG backend for better RTSP support (default: True)
            rtsp_transport: RTSP transport protocol - "tcp" or "udp" (default: "tcp")
            filter_chain: Optional FilterChain instance to apply to frames
        """
        super().__init__(cfg, filter_chain=filter_chain)
        self.rtsp_url = rtsp_url or (cfg.camera_source if isinstance(cfg.camera_source, str) else None)
        if not self.rtsp_url:
            raise ValueError("RTSP URL must be provided either as rtsp_url parameter or in cfg.camera_source")
        
        # Ensure URL starts with rtsp://
        if not self.rtsp_url.startswith("rtsp://"):
            raise ValueError(f"Invalid RTSP URL: {self.rtsp_url}. Must start with 'rtsp://'")
        
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_attempts = 0
        self.use_ffmpeg_backend = use_ffmpeg_backend
        self.rtsp_transport = rtsp_transport
        
        self.cap = None
        self.running = False
        self.frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.connected = False
        self.last_frame_time = None
        
    def _build_rtsp_url(self) -> str:
        """
        Build RTSP URL with optional transport parameters.
        
        Returns:
            RTSP URL with transport options if needed
        """
        url = self.rtsp_url
        
        # Add transport parameter if using FFMPEG backend
        if self.use_ffmpeg_backend and self.rtsp_transport:
            # FFMPEG uses different URL format for transport
            # For OpenCV with FFMPEG, we can add it as a parameter
            if "?" not in url:
                url += f"?rtsp_transport={self.rtsp_transport}"
            elif "rtsp_transport" not in url:
                url += f"&rtsp_transport={self.rtsp_transport}"
        
        return url
    
    def _connect(self) -> bool:
        """
        Attempt to connect to the RTSP stream.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            url = self._build_rtsp_url()
            logger.info(f"Connecting to RTSP stream: {self.rtsp_url}")
            
            # Use FFMPEG backend for better RTSP support if requested
            if self.use_ffmpeg_backend:
                # Try FFMPEG backend first (better RTSP support)
                self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            else:
                # Use default backend
                self.cap = cv2.VideoCapture(url)
            
            if not self.cap.isOpened():
                logger.warning("Failed to open RTSP stream")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                return False
            
            # Configure for low latency (important for RTSP streams)
            # Set buffer size to 1 to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set FPS if configured
            if self.cfg.fps:
                self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
            
            # Set timeout for frame reading (helps detect connection issues faster)
            # Note: This may not work on all backends, but worth trying
            
            # Try to read a frame to verify connection
            # Give it a few attempts as RTSP can be slow to start
            for attempt in range(3):
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    self.connected = True
                    self.reconnect_attempts = 0
                    width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    logger.info(f"Connected successfully - Resolution: {width}x{height}, FPS: {fps}")
                    return True
                time.sleep(0.5)
            
            logger.warning("Connection failed: unable to read valid frame after multiple attempts")
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
        Attempt to reconnect to the RTSP stream.
        
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
        logger.info(f"Starting RTSP capture from {self.rtsp_url}")
        
        # Initial connection
        if not self._connect():
            logger.error("Initial connection failed")
            return
        
        self.running = True
        consecutive_failures = 0
        max_consecutive_failures = 10  # Number of consecutive read failures before attempting reconnect
        last_successful_frame_time = time.time()
        connection_timeout = 30.0  # Consider connection lost if no frames for 30 seconds
        
        while self.running:
            try:
                start_time = time.perf_counter()
                ret, frame = self.cap.read()
                self.last_frame_time = time.time()
                
                if not ret or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    
                    # Check for connection timeout
                    time_since_last_frame = time.time() - last_successful_frame_time
                    if time_since_last_frame > connection_timeout:
                        logger.warning(f"No frames received for {time_since_last_frame:.1f} seconds, attempting reconnect...")
                        if not self._reconnect():
                            if self.max_reconnect_attempts > 0 and self.reconnect_attempts >= self.max_reconnect_attempts:
                                logger.error("Stopping due to max reconnection attempts")
                                break
                            time.sleep(self.reconnect_delay)
                            continue
                        consecutive_failures = 0
                        last_successful_frame_time = time.time()
                    elif consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"{consecutive_failures} consecutive read failures, attempting reconnect...")
                        if not self._reconnect():
                            if self.max_reconnect_attempts > 0 and self.reconnect_attempts >= self.max_reconnect_attempts:
                                logger.error("Stopping due to max reconnection attempts")
                                break
                            time.sleep(self.reconnect_delay)
                            continue
                        consecutive_failures = 0
                        last_successful_frame_time = time.time()
                    else:
                        time.sleep(0.1)
                    continue
                
                # Successful frame read
                consecutive_failures = 0
                last_successful_frame_time = time.time()
                
                # Apply filter chain if provided
                if self.filter_chain:
                    frame = self.filter_chain.apply(frame)
                
                with self.lock:
                    self.frame = frame.copy()

                delta_time = time.perf_counter() - start_time
                # logger.debug(f"capture time: {delta_time:0.5f}")

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
        logger.info("RTSP camera capture stopped")
    
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

    def get_last_frame_time(self) -> float:
        return self.last_frame_time

