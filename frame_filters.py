"""
Frame filter system for processing video frames.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameFilter(ABC):
    """Base class for frame filters."""
    
    @abstractmethod
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply filter to a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Filtered frame (BGR format)
        """
        pass


class FilterChain:
    """Chain of filters to apply to frames sequentially."""
    
    def __init__(self):
        self.filters: List[FrameFilter] = []
    
    def add_filter(self, filter_obj: FrameFilter):
        """Add a filter to the chain."""
        self.filters.append(filter_obj)
        logger.debug(f"Added filter {filter_obj.__class__.__name__} to chain")
    
    def remove_filter(self, filter_obj: FrameFilter):
        """Remove a filter from the chain."""
        if filter_obj in self.filters:
            self.filters.remove(filter_obj)
            logger.debug(f"Removed filter {filter_obj.__class__.__name__} from chain")
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply all filters in the chain sequentially.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Filtered frame (BGR format)
        """
        result = frame.copy()
        for filter_obj in self.filters:
            result = filter_obj.apply(result)
        return result


class TextOverlayFilter(FrameFilter):
    """Filter that adds text overlay to frames."""
    
    def __init__(
        self,
        text: str,
        position: Tuple[int, int],
        font_scale: float = 1.0,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        transparency: float = 1.0,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        background: bool = True,
        background_color: Tuple[int, int, int] = (0, 0, 0),
        background_transparency: float = 0.7,
    ):
        """
        Initialize text overlay filter.
        
        Args:
            text: Text to display
            position: (x, y) position of text (top-left corner)
            font_scale: Font size scale
            color: Text color in BGR format (B, G, R)
            thickness: Text thickness
            transparency: Text transparency (0.0 to 1.0, where 1.0 is opaque)
            font: OpenCV font constant
            background: Whether to draw background rectangle
            background_color: Background color in BGR format
            background_transparency: Background transparency (0.0 to 1.0)
        """
        self.text = text
        self.position = position
        self.font_scale = font_scale
        self.color = color
        self.thickness = thickness
        self.transparency = transparency
        self.font = font
        self.background = background
        self.background_color = background_color
        self.background_transparency = background_transparency
    
    def set_text(self, text: str):
        """Update the text to display."""
        self.text = text
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply text overlay to frame."""
        if not self.text:
            return frame
        
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # Get text size to calculate background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            self.text, self.font, self.font_scale, self.thickness
        )
        
        x, y = self.position
        
        # Draw background rectangle if enabled
        if self.background:
            # Calculate background rectangle coordinates
            padding = 5
            bg_x1 = max(0, x - padding)
            bg_y1 = max(0, y - text_height - padding)
            bg_x2 = min(w, x + text_width + padding)
            bg_y2 = min(h, y + baseline + padding)
            
            # Create overlay for background
            overlay = result.copy()
            cv2.rectangle(
                overlay,
                (bg_x1, bg_y1),
                (bg_x2, bg_y2),
                self.background_color,
                -1
            )
            # Blend background with transparency
            cv2.addWeighted(overlay, self.background_transparency, result, 1 - self.background_transparency, 0, result)
        
        # Draw text with transparency
        if self.transparency < 1.0:
            # Create overlay for text
            overlay = result.copy()
            cv2.putText(
                overlay,
                self.text,
                (x, y),
                self.font,
                self.font_scale,
                self.color,
                self.thickness,
                cv2.LINE_AA
            )
            # Blend text with transparency
            cv2.addWeighted(overlay, self.transparency, result, 1 - self.transparency, 0, result)
        else:
            # Draw text directly if fully opaque
            cv2.putText(
                result,
                self.text,
                (x, y),
                self.font,
                self.font_scale,
                self.color,
                self.thickness,
                cv2.LINE_AA
            )
        
        return result


class WeatherTextOverlayFilter(TextOverlayFilter):
    """Text overlay filter that dynamically updates text from weather service."""
    
    def __init__(
        self,
        weather_service,
        position = "top_right",  # Can be "top_right" or Tuple[int, int]
        font_scale: float = 0.7,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        transparency: float = 1.0,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        background: bool = True,
        background_color: Tuple[int, int, int] = (0, 0, 0),
        background_transparency: float = 0.7,
    ):
        """
        Initialize weather text overlay filter.
        
        Args:
            weather_service: WeatherService instance to get temperature from
            position: (x, y) position of text (top-left corner), or "top_right" for auto-positioning
            font_scale: Font size scale
            color: Text color in BGR format (B, G, R)
            thickness: Text thickness
            transparency: Text transparency (0.0 to 1.0, where 1.0 is opaque)
            font: OpenCV font constant
            background: Whether to draw background rectangle
            background_color: Background color in BGR format
            background_transparency: Background transparency (0.0 to 1.0)
        """
        # Set initial position - will be overridden if auto_position
        initial_position = position if isinstance(position, tuple) else (10, 10)
        super().__init__(
            text="Loading...",
            position=initial_position,
            font_scale=font_scale,
            color=color,
            thickness=thickness,
            transparency=transparency,
            font=font,
            background=background,
            background_color=background_color,
            background_transparency=background_transparency,
        )
        self.weather_service = weather_service
        self.auto_position = position == "top_right"
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply weather text overlay to frame with dynamic updates."""
        # Update text from weather service
        if self.weather_service:
            temp_str = self.weather_service.get_temperature_string()
            self.set_text(f"{temp_str}")
        
        # Auto-position to top right if requested
        if self.auto_position:
            height, width = frame.shape[:2]
            (text_width, text_height), _ = cv2.getTextSize(
                self.text,
                self.font,
                self.font_scale,
                self.thickness
            )
            padding = 10
            self.position = (width - text_width - padding, 30)
        
        # Apply parent text overlay
        return super().apply(frame)

