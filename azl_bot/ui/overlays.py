"""Overlay rendering for the UI."""

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


class OverlayRenderer:
    """Renders overlays on captured frames for UI display."""
    
    def __init__(self) -> None:
        """Initialize overlay renderer."""
        self.colors = {
            "region": (100, 100, 255),     # Blue
            "candidate": (0, 255, 255),    # Cyan
            "tap": (255, 0, 255),          # Magenta
            "success": (0, 255, 0),        # Green
            "failure": (0, 0, 255),        # Red
            "text": (255, 255, 255),       # White
            "box": (255, 255, 0)           # Yellow
        }
    
    def render_overlays(self, image: np.ndarray, overlay_data: Dict[str, Any]) -> np.ndarray:
        """Render overlays on image.
        
        Args:
            image: Input image (RGB format)
            overlay_data: Dictionary containing overlay information
            
        Returns:
            Image with overlays rendered
        """
        # Work on a copy
        overlay_img = image.copy()
        h, w = overlay_img.shape[:2]
        
        # Draw regions if provided
        if "regions" in overlay_data and overlay_data["regions"]:
            self._draw_regions(overlay_img, overlay_data["regions"])
        
        # Draw candidates if provided
        if "candidates" in overlay_data and overlay_data["candidates"]:
            self._draw_candidates(overlay_img, overlay_data["candidates"])
        
        # Draw last tap point if provided
        if "last_tap" in overlay_data and overlay_data["last_tap"]:
            self._draw_tap_point(overlay_img, overlay_data["last_tap"])
        
        # Draw bounding boxes if provided
        if "boxes" in overlay_data and overlay_data["boxes"]:
            self._draw_boxes(overlay_img, overlay_data["boxes"])
        
        # Draw OCR results if provided
        if "ocr_results" in overlay_data and overlay_data["ocr_results"]:
            self._draw_ocr_results(overlay_img, overlay_data["ocr_results"])
        
        return overlay_img
    
    def _draw_regions(self, img: np.ndarray, regions: Dict[str, Tuple[float, float, float, float]]) -> None:
        """Draw region rectangles on image.
        
        Args:
            img: Image to draw on (RGB format)
            regions: Dictionary of region name to (x,y,w,h) tuples
        """
        h, w = img.shape[:2]
        
        for region_name, (rx, ry, rw, rh) in regions.items():
            # Convert normalized to pixels
            x1 = int(rx * w)
            y1 = int(ry * h)
            x2 = int((rx + rw) * w)
            y2 = int((ry + rh) * h)
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), self.colors["region"], 2)
            
            # Draw label with background
            label = region_name
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Background rectangle for text
            cv2.rectangle(img, (x1, y1 - text_h - baseline - 2), (x1 + text_w + 4, y1), self.colors["region"], -1)
            
            # Text
            cv2.putText(img, label, (x1 + 2, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["text"], 1)
    
    def _draw_candidates(self, img: np.ndarray, candidates: List[Dict[str, Any]]) -> None:
        """Draw candidate points on image.
        
        Args:
            img: Image to draw on (RGB format)
            candidates: List of candidate data
        """
        h, w = img.shape[:2]
        
        for i, candidate in enumerate(candidates):
            if "point" not in candidate or "confidence" not in candidate:
                continue
            
            x, y = candidate["point"]
            conf = candidate["confidence"]
            method = candidate.get("method", "unknown")
            
            # Convert to pixels
            px = int(x * w)
            py = int(y * h)
            
            # Choose color based on confidence
            if conf >= 0.8:
                color = self.colors["success"]
            elif conf >= 0.6:
                color = self.colors["candidate"]
            else:
                color = self.colors["failure"]
            
            # Draw circle
            cv2.circle(img, (px, py), 8, color, 2)
            cv2.circle(img, (px, py), 2, self.colors["text"], -1)  # Center dot
            
            # Draw label
            label = f"{method}:{conf:.2f}"
            cv2.putText(img, label, (px + 12, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _draw_tap_point(self, img: np.ndarray, tap_data: Dict[str, Any]) -> None:
        """Draw last tap point on image.
        
        Args:
            img: Image to draw on (RGB format)
            tap_data: Tap point data
        """
        if "point" not in tap_data:
            return
        
        h, w = img.shape[:2]
        x, y = tap_data["point"]
        
        # Convert to pixels
        px = int(x * w)
        py = int(y * h)
        
        # Check if tap is recent (fade effect)
        tap_time = tap_data.get("timestamp", 0)
        current_time = time.time()
        age = current_time - tap_time
        
        if age > 5.0:  # Don't show taps older than 5 seconds
            return
        
        # Fade alpha based on age
        alpha = max(0.3, 1.0 - (age / 5.0))
        
        # Draw large circle for tap
        color = tuple(int(c * alpha) for c in self.colors["tap"])
        
        cv2.circle(img, (px, py), 20, color, 3)
        cv2.circle(img, (px, py), 4, self.colors["text"], -1)  # White center
        
        # Draw timestamp if available
        if "timestamp" in tap_data:
            label = f"TAP @{age:.1f}s ago"
            cv2.putText(img, label, (px + 25, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _draw_boxes(self, img: np.ndarray, boxes: List[Dict[str, Any]]) -> None:
        """Draw bounding boxes on image.
        
        Args:
            img: Image to draw on (RGB format)
            boxes: List of box data
        """
        h, w = img.shape[:2]
        
        for box_data in boxes:
            if "bbox" not in box_data:
                continue
            
            bx, by, bw, bh = box_data["bbox"]
            
            # Convert to pixels
            x1 = int(bx * w)
            y1 = int(by * h)
            x2 = int((bx + bw) * w)
            y2 = int((by + bh) * h)
            
            # Draw rectangle
            color = self.colors["box"]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label if available
            if "label" in box_data:
                label = box_data["label"]
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _draw_ocr_results(self, img: np.ndarray, ocr_results: List[Dict[str, Any]]) -> None:
        """Draw OCR results on image.
        
        Args:
            img: Image to draw on (RGB format)
            ocr_results: List of OCR results
        """
        h, w = img.shape[:2]
        
        for result in ocr_results:
            if "box_norm" not in result or "text" not in result:
                continue
            
            box = result["box_norm"]
            text = result["text"]
            conf = result.get("conf", 1.0)
            
            # Convert to pixels
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int((box[0] + box[2]) * w)
            y2 = int((box[1] + box[3]) * h)
            
            # Choose color based on confidence
            if conf >= 0.8:
                color = self.colors["success"]
            elif conf >= 0.6:
                color = self.colors["candidate"]
            else:
                color = self.colors["failure"]
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            
            # Draw text with background (for readability)
            if len(text) > 0:
                font_scale = 0.4
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Background rectangle for text
                bg_x1 = x1
                bg_y1 = y1 - text_h - baseline - 2
                bg_x2 = x1 + text_w + 4
                bg_y2 = y1
                
                cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)  # Black background
                
                # Text
                cv2.putText(img, text, (x1 + 2, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def create_overlay_data(regions: Optional[Dict] = None, candidates: Optional[List] = None,
                       last_tap: Optional[Dict] = None, boxes: Optional[List] = None,
                       ocr_results: Optional[List] = None) -> Dict[str, Any]:
    """Create overlay data dictionary.
    
    Args:
        regions: Region definitions
        candidates: Candidate points
        last_tap: Last tap point data
        boxes: Bounding boxes
        ocr_results: OCR results
        
    Returns:
        Overlay data dictionary
    """
    overlay_data = {}
    
    if regions:
        overlay_data["regions"] = regions
    
    if candidates:
        overlay_data["candidates"] = candidates
    
    if last_tap:
        overlay_data["last_tap"] = last_tap
    
    if boxes:
        overlay_data["boxes"] = boxes
    
    if ocr_results:
        overlay_data["ocr_results"] = ocr_results
    
    return overlay_data