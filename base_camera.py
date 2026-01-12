"""
Base camera interface for all camera implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import threading

from config import AppConfig

logger = logging.getLogger(__name__)


class BaseCameraCapture(threading.Thread, ABC):
    """
    Abstract base class for all camera capture implementations.
    
    This class defines the common interface that all camera implementations
    must follow, allowing the Detector to work with any camera type.
    """
    
    def __init__(self, cfg: AppConfig):
        """
        Initialize the base camera capture.
        
        Args:
            cfg: Application configuration
        """
        super().__init__(daemon=True)
        self.cfg = cfg
        self.running = False
    
    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest captured frame.
        
        Returns:
            Latest frame as numpy array, or None if no frame available
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """
        Stop the capture thread.
        """
        pass
    
    @abstractmethod
    def run(self) -> None:
        """
        Main capture loop. This method is called when the thread starts.
        Should be implemented by subclasses to handle frame capture.
        """
        pass

