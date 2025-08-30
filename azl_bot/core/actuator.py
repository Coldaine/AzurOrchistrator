"""Actuator for performing taps and swipes on Android device."""

import time
from typing import Literal

from loguru import logger

from .device import Device


class Actuator:
    """Handles input actions on Android device."""
    
    def __init__(self, device: Device, backend: Literal["adb", "minitouch"] = "adb") -> None:
        """Initialize actuator with device and backend.
        
        Args:
            device: Device instance for ADB communication
            backend: Input backend ("adb" or "minitouch")
        """
        self.device = device
        self.backend = backend
        self._last_tap_time = 0.0
        self._last_tap_x = 0.0 
        self._last_tap_y = 0.0
        
        if backend == "minitouch":
            logger.warning("minitouch backend not yet implemented, falling back to adb")
            self.backend = "adb"
    
    def tap_norm(self, x: float, y: float) -> None:
        """Tap at normalized coordinates.
        
        Args:
            x: X coordinate (0.0 to 1.0)
            y: Y coordinate (0.0 to 1.0)
        """
        # Debounce duplicate taps
        current_time = time.time()
        if (current_time - self._last_tap_time < 0.5 and 
            abs(x - self._last_tap_x) < 0.01 and 
            abs(y - self._last_tap_y) < 0.01):
            logger.debug(f"Debouncing duplicate tap at ({x:.3f}, {y:.3f})")
            return
            
        # Convert normalized to pixel coordinates
        info = self.device.info()
        pixel_x = int(x * info["width"])
        pixel_y = int(y * info["height"])
        
        # Clamp to screen bounds
        pixel_x = max(0, min(pixel_x, info["width"] - 1))
        pixel_y = max(0, min(pixel_y, info["height"] - 1))
        
        logger.info(f"Tapping at norm({x:.3f}, {y:.3f}) -> pixel({pixel_x}, {pixel_y})")
        
        if self.backend == "adb":
            self.device._adb("shell", "input", "tap", str(pixel_x), str(pixel_y))
        
        # Update debounce tracking
        self._last_tap_time = current_time
        self._last_tap_x = x
        self._last_tap_y = y
        
        time.sleep(0.3)  # Brief delay after tap
    
    def swipe_norm(self, x1: float, y1: float, x2: float, y2: float, ms: int = 200) -> None:
        """Swipe between normalized coordinates.
        
        Args:
            x1: Start X coordinate (0.0 to 1.0)
            y1: Start Y coordinate (0.0 to 1.0) 
            x2: End X coordinate (0.0 to 1.0)
            y2: End Y coordinate (0.0 to 1.0)
            ms: Swipe duration in milliseconds
        """
        info = self.device.info()
        
        # Convert to pixel coordinates
        pixel_x1 = int(x1 * info["width"])
        pixel_y1 = int(y1 * info["height"])
        pixel_x2 = int(x2 * info["width"])
        pixel_y2 = int(y2 * info["height"])
        
        # Clamp to screen bounds
        pixel_x1 = max(0, min(pixel_x1, info["width"] - 1))
        pixel_y1 = max(0, min(pixel_y1, info["height"] - 1))
        pixel_x2 = max(0, min(pixel_x2, info["width"] - 1))
        pixel_y2 = max(0, min(pixel_y2, info["height"] - 1))
        
        logger.info(f"Swiping from norm({x1:.3f}, {y1:.3f}) to norm({x2:.3f}, {y2:.3f}) "
                   f"-> pixel({pixel_x1}, {pixel_y1}) to ({pixel_x2}, {pixel_y2}) in {ms}ms")
        
        if self.backend == "adb":
            self.device._adb("shell", "input", "swipe", 
                           str(pixel_x1), str(pixel_y1), 
                           str(pixel_x2), str(pixel_y2), 
                           str(ms))
        
        time.sleep(max(0.5, ms / 1000.0 + 0.2))  # Wait for swipe completion
        
    def tap_pixel(self, x: int, y: int) -> None:
        """Tap at pixel coordinates (for internal use).
        
        Args:
            x: X pixel coordinate
            y: Y pixel coordinate  
        """
        info = self.device.info()
        norm_x = x / info["width"]
        norm_y = y / info["height"]
        self.tap_norm(norm_x, norm_y)