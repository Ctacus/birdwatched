"""
Weather service for fetching weather data from public APIs.
"""

import logging
import os
import time
from typing import Optional, Tuple
from threading import Lock

import requests

logger = logging.getLogger(__name__)


class WeatherService:
    """Service for fetching weather data from Open-Meteo API."""
    
    # Open-Meteo API endpoint (free, no API key required)
    API_BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    def __init__(
        self,
        latitude: float = 51.5074,  # Default: London
        longitude: float = -0.1278,
        units: str = "metric"
    ):
        """
        Initialize weather service.
        
        Args:
            latitude: Latitude coordinate (default: 51.5074 for London)
            longitude: Longitude coordinate (default: -0.1278 for London)
            units: Temperature units ("metric" for Celsius, "imperial" for Fahrenheit)
        """
        self.latitude = latitude
        self.longitude = longitude
        self.units = units

        self._temperature: Optional[float] = None
        self._description: Optional[str] = None
        self._last_update: float = 0.0
        self._lock = Lock()
        self._error_count = 0
    
    def _fetch_weather(self) -> Optional[dict]:
        """Fetch weather data from Open-Meteo API."""
        try:
            params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "hourly": "temperature_2m",
                "timezone": "auto",
                "forecast_days": 1,
                "wind_speed_unit": "ms",
                "forecast_hours": 1,
            }
            
            logger.debug(f"Fetching weather data for {self.latitude}, {self.longitude}")
            response = requests.get(self.API_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            self._error_count = 0
            return data
        
        except requests.exceptions.RequestException as e:
            self._error_count += 1
            logger.warning(f"Failed to fetch weather data: {e} (error count: {self._error_count})")
            return None
    
    def update(self) -> bool:
        """
        Update weather data if enough time has passed.
        
        Returns:
            True if update was successful, False otherwise
        """
        current_time = time.time()
        logger.debug(f"Updating weather data...")
        
        with self._lock:
            # Fetch new data
            data = self._fetch_weather()
            if data:
                try:
                    # Extract temperature from hourly data (first hour)
                    hourly = data.get("hourly", {})
                    temperatures = hourly.get("temperature_2m", [])
                    if temperatures and len(temperatures) > 0:
                        self._temperature = temperatures[0]
                        self._last_update = current_time
                        logger.info(f"Weather updated: {self._temperature:.1f}°C")
                        return True
                    else:
                        logger.warning("No temperature data in API response")
                        return False
                except (KeyError, IndexError, TypeError) as e:
                    logger.error(f"Unexpected API response format: {e}")
                    return False
            else:
                return False
    
    def get_temperature(self) -> Optional[float]:
        """Get current temperature."""
        return self._temperature
    
    def get_description(self) -> Optional[str]:
        """Get current weather description."""
        with self._lock:
            return self._description
    
    def get_temperature_string(self) -> str:
        """
        Get formatted temperature string.
        
        Returns:
            Formatted string like "22.5C" or "N/A" if unavailable
            Note: Uses "C" instead of "°C" because OpenCV can't render Unicode degree symbol
        """
        temp = self.get_temperature()
        if temp is not None:
            unit = "C" if self.units == "metric" else "F"
            return f"{temp:.0f}{unit}"
        return ""
    
    def set_location(self, latitude: float, longitude: float):
        """Update location coordinates and reset cache."""
        with self._lock:
            self.latitude = latitude
            self.longitude = longitude
            self._temperature = None
            self._last_update = 0.0
            logger.info(f"Location updated to: {latitude}, {longitude}")


class WeatherScheduler:
    """Scheduler for periodic weather updates."""
    
    def __init__(
        self,
        weather_service: WeatherService,
        update_interval: float = 300.0,  # 5 minutes
    ):
        """
        Initialize weather scheduler.
        
        Args:
            weather_service: WeatherService instance
            update_interval: Update interval in seconds (default: 5 minutes)
        """
        self.weather_service = weather_service
        self.update_interval = update_interval
        self.running = False
        self._thread = None
    
    def start(self):
        """Start the scheduler thread."""
        if self.running:
            return
        
        self.running = True
        import threading
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Weather scheduler started (update interval: {self.update_interval}s)")
        

    def stop(self):
        """Stop the scheduler thread."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("Weather scheduler stopped")
    
    def _run(self):
        """Main scheduler loop."""
        while self.running:            
            self.weather_service.update()
            time.sleep(self.update_interval)

