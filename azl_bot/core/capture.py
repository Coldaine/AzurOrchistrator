"""Screen capture and preprocessing."""

import time
from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from loguru import logger

from .device import Device


@dataclass
class Frame:
    """Captured frame with metadata."""
    png_bytes: bytes
    image_bgr: np.ndarray  # cropped to active area (no letterbox)
    full_w: int
    full_h: int  
    active_rect: tuple[int, int, int, int]  # (x,y,w,h) in full pixels
    ts: float


class Capture:
    """Screen capture and preprocessing."""
    
    def __init__(self, device: Device) -> None:
        """Initialize capture with device.
        
        Args:
            device: Device instance for screen capture
        """
        self.device = device
        
    def grab(self) -> Frame:
        """Capture current screen frame.
        
        Returns:
            Frame with processed image and metadata
        """
        ts = time.time()
        logger.debug("Capturing screen frame")
        
        # Get raw PNG from device
        png_bytes = self.device.screencap_png()
        
        # Convert PNG to OpenCV image
        pil_image = Image.open(BytesIO(png_bytes))
        image_rgb = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        full_h, full_w = image_bgr.shape[:2]
        logger.debug(f"Full frame size: {full_w}x{full_h}")
        
        # Detect letterboxing
        active_rect = self.detect_letterbox(image_bgr)
        logger.debug(f"Active area: {active_rect}")
        
        # Crop to active area
        x, y, w, h = active_rect
        image_bgr_cropped = image_bgr[y:y+h, x:x+w]
        
        return Frame(
            png_bytes=png_bytes,
            image_bgr=image_bgr_cropped,
            full_w=full_w,
            full_h=full_h,
            active_rect=active_rect,
            ts=ts
        )
    
    def detect_letterbox(self, img_bgr: np.ndarray) -> tuple[int, int, int, int]:
        """Detect letterbox borders and return active area.
        
        Args:
            img_bgr: Full screen image in BGR format
            
        Returns:
            (x, y, w, h) tuple of active area in pixels
        """
        h, w = img_bgr.shape[:2]
        
        # Convert to grayscale for border detection
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Check top and bottom borders for letterboxing
        top_border = 0
        bottom_border = h
        left_border = 0
        right_border = w
        
        # Scan from top
        for y in range(min(h // 4, 100)):  # Check up to 1/4 height or 100px
            row = gray[y, :]
            if self._is_letterbox_border(row):
                top_border = y + 1
            else:
                break
                
        # Scan from bottom  
        for y in range(h - 1, max(3 * h // 4, h - 100), -1):
            row = gray[y, :]
            if self._is_letterbox_border(row):
                bottom_border = y
            else:
                break
                
        # Scan from left
        for x in range(min(w // 4, 100)):
            col = gray[:, x]
            if self._is_letterbox_border(col):
                left_border = x + 1
            else:
                break
                
        # Scan from right
        for x in range(w - 1, max(3 * w // 4, w - 100), -1):
            col = gray[:, x]
            if self._is_letterbox_border(col):
                right_border = x
            else:
                break
        
        # Ensure we have a valid rectangle
        active_w = right_border - left_border
        active_h = bottom_border - top_border
        
        if active_w <= 0 or active_h <= 0:
            logger.warning("Invalid letterbox detection, using full frame")
            return (0, 0, w, h)
            
        # Sanity check - active area should be reasonable size
        if active_w < w * 0.5 or active_h < h * 0.5:
            logger.warning("Letterbox detection found very small active area, using full frame")
            return (0, 0, w, h)
            
        return (left_border, top_border, active_w, active_h)
    
    def _is_letterbox_border(self, line: np.ndarray) -> bool:
        """Check if a line appears to be a letterbox border.
        
        Args:
            line: 1D array representing a row or column
            
        Returns:
            True if line appears to be a letterbox border
        """
        # Check if line is mostly uniform (low variance)
        if len(line) == 0:
            return False
            
        # Calculate variance 
        mean_val = np.mean(line)
        variance = np.var(line)
        
        # Letterbox borders are typically dark and uniform
        is_dark = mean_val < 30  # Dark threshold
        is_uniform = variance < 100  # Low variance threshold
        
        return is_dark and is_uniform
        
    def norm_to_pixels(self, x_norm: float, y_norm: float, frame: Frame) -> tuple[int, int]:
        """Convert normalized coordinates to pixels in full frame.
        
        Args:
            x_norm: Normalized X coordinate (0.0 to 1.0) 
            y_norm: Normalized Y coordinate (0.0 to 1.0)
            frame: Frame containing active area info
            
        Returns:
            (x, y) pixel coordinates in full frame
        """
        ax, ay, aw, ah = frame.active_rect
        
        # Convert normalized to active area pixels
        active_x = int(x_norm * aw)
        active_y = int(y_norm * ah)
        
        # Convert to full frame pixels
        full_x = ax + active_x
        full_y = ay + active_y
        
        return (full_x, full_y)
        
    def pixels_to_norm(self, x: int, y: int, frame: Frame) -> tuple[float, float]:
        """Convert full frame pixels to normalized coordinates.
        
        Args:
            x: X pixel coordinate in full frame
            y: Y pixel coordinate in full frame  
            frame: Frame containing active area info
            
        Returns:
            (x_norm, y_norm) normalized coordinates
        """
        ax, ay, aw, ah = frame.active_rect
        
        # Convert to active area coordinates
        active_x = x - ax
        active_y = y - ay
        
        # Convert to normalized 
        x_norm = active_x / aw if aw > 0 else 0.0
        y_norm = active_y / ah if ah > 0 else 0.0
        
        return (x_norm, y_norm)