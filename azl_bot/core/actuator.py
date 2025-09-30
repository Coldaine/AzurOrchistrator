
import time
from typing import Literal, Optional

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
        self.capture = None  # Will be set by bootstrap
        self._last_tap_time = 0.0
        self._last_tap_x = 0.0
        self._last_tap_y = 0.0

        if backend == "minitouch":
            logger.warning("minitouch backend not yet implemented, falling back to adb")
            self.backend = "adb"

    def _normalize_to_device(self, x: float, y: float, active_rect: Optional[tuple[int, int, int, int]] = None) -> tuple[int, int]:
        """Convert normalized (0-1) coordinates to device coordinates.
        
        CRITICAL: Uses active_area bounds, not full device dimensions!
        """
        # Use provided active_rect or get from capture component
        if active_rect is None and hasattr(self, 'capture') and self.capture:
            # Get the most recent frame's active_rect
            try:
                # This assumes capture has a way to get the latest active_rect
                # In practice, this would need to be implemented in capture
                active_rect = getattr(self.capture, 'last_active_rect', None)
            except AttributeError:
                pass

        if active_rect:
            ax, ay, aw, ah = active_rect
            device_x = int(ax + x * aw)
            device_y = int(ay + y * ah)
            return device_x, device_y

        # Fallback to full device dimensions
        info = self.device.info()
        device_x = int(x * info["width"])
        device_y = int(y * info["height"])
        logger.warning("No active_rect available, using full device coordinates")
        return device_x, device_y

    def tap_norm(self, x: float, y: float, active_rect: tuple[int, int, int, int] | None = None) -> None:
        """Tap at normalized coordinates.

        Args:
            x: X coordinate (0.0 to 1.0)
            y: Y coordinate (0.0 to 1.0)
            active_rect: Active area (x, y, w, h) in pixels, if None uses full device
        """
        # Debounce duplicate taps
        current_time = time.time()
        if (current_time - self._last_tap_time < 0.5 and
            abs(x - self._last_tap_x) < 0.01 and
            abs(y - self._last_tap_y) < 0.01):
            logger.debug(f"Debouncing duplicate tap at ({x:.3f}, {y:.3f})")
            return

        # Convert normalized to pixel coordinates
        if active_rect is not None:
            # Use provided active area for coordinate transformation
            ax, ay, aw, ah = active_rect
            pixel_x = int(ax + x * aw)
            pixel_y = int(ay + y * ah)
            logger.info(f"Tapping at norm({x:.3f}, {y:.3f}) -> active({pixel_x}, {pixel_y})")
        else:
            # Use _normalize_to_device method which tries to get active_rect from capture
            pixel_x, pixel_y = self._normalize_to_device(x, y)

        # Clamp to screen bounds
        info = self.device.info()
        pixel_x = max(0, min(pixel_x, info["width"] - 1))
        pixel_y = max(0, min(pixel_y, info["height"] - 1))

        logger.info(f"Final tap coordinates: pixel({pixel_x}, {pixel_y})")

        if self.backend == "adb":
            self.device._adb("shell", "input", "tap", str(pixel_x), str(pixel_y))

        # Update debounce tracking
        self._last_tap_time = current_time
        self._last_tap_x = x
        self._last_tap_y = y

        # Mark activity for frame rate management
        if hasattr(self, 'capture') and self.capture and hasattr(self.capture, 'frame_rate_manager'):
            self.capture.frame_rate_manager.mark_activity()

        time.sleep(0.3)  # Brief delay after tap

    def swipe_norm(self, x1: float, y1: float, x2: float, y2: float, ms: int = 200, active_rect: tuple[int, int, int, int] | None = None) -> None:
        """Swipe between normalized coordinates.

        Args:
            x1: Start X coordinate (0.0 to 1.0)
            y1: Start Y coordinate (0.0 to 1.0)
            x2: End X coordinate (0.0 to 1.0)
            y2: End Y coordinate (0.0 to 1.0)
            ms: Swipe duration in milliseconds
            active_rect: Active area (x, y, w, h) in pixels, if None uses full device
        """
        # Convert to pixel coordinates
        if active_rect is not None:
            # Use active area for coordinate transformation
            ax, ay, aw, ah = active_rect
            pixel_x1 = int(ax + x1 * aw)
            pixel_y1 = int(ay + y1 * ah)
            pixel_x2 = int(ax + x2 * aw)
            pixel_y2 = int(ay + y2 * ah)
            logger.info(f"Swiping from norm({x1:.3f}, {y1:.3f}) to norm({x2:.3f}, {y2:.3f}) "
                       f"-> active({pixel_x1}, {pixel_y1}) to ({pixel_x2}, {pixel_y2}) in {ms}ms")
        else:
            # Use _normalize_to_device method which tries to get active_rect from capture
            pixel_x1, pixel_y1 = self._normalize_to_device(x1, y1)
            pixel_x2, pixel_y2 = self._normalize_to_device(x2, y2)

        # Clamp to screen bounds
        info = self.device.info()
        pixel_x1 = max(0, min(pixel_x1, info["width"] - 1))
        pixel_y1 = max(0, min(pixel_y1, info["height"] - 1))
        pixel_x2 = max(0, min(pixel_x2, info["width"] - 1))
        pixel_y2 = max(0, min(pixel_y2, info["height"] - 1))

        logger.info(f"Final swipe coordinates: pixel({pixel_x1}, {pixel_y1}) to ({pixel_x2}, {pixel_y2}) in {ms}ms")

        if self.backend == "adb":
            self.device._adb("shell", "input", "swipe",
                           str(pixel_x1), str(pixel_y1),
                           str(pixel_x2), str(pixel_y2),
                           str(ms))

        # Mark activity for frame rate management
        if hasattr(self, 'capture') and self.capture and hasattr(self.capture, 'frame_rate_manager'):
            self.capture.frame_rate_manager.mark_activity()

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