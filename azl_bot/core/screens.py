"""Screen definitions and region helpers."""

from typing import List, Tuple

import cv2
import numpy as np
from loguru import logger

from .capture import Frame


class Regions:
    """Predefined screen regions in normalized coordinates."""
    top_bar = (0.00, 0.00, 1.00, 0.12)
    bottom_bar = (0.00, 0.85, 1.00, 0.15)
    left_panel = (0.00, 0.12, 0.20, 0.73)
    center = (0.20, 0.12, 0.60, 0.73)
    right_panel = (0.80, 0.12, 0.20, 0.73)


def expected_home_elements() -> List[str]:
    """Return list of UI elements expected on the home screen.
    
    Returns:
        List of element names that should be present on home screen
    """
    return ["Commissions", "Missions", "Mailbox"]


def detect_red_badges(frame: Frame, icon_anchors: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Detect red notification badges near known icon locations.
    
    Args:
        frame: Current screen frame
        icon_anchors: List of (x, y) normalized coordinates where badges might appear
        
    Returns:
        List of (x, y) normalized coordinates where badges were detected
    """
    badges = []
    
    # Convert to HSV for red detection
    hsv = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    
    # Define red color ranges in HSV
    # Red hue wraps around, so we need two ranges
    lower_red1 = np.array([0, 153, 102])    # Hue 0-10, S≥0.6*255, V≥0.4*255
    upper_red1 = np.array([10, 255, 255])
    
    lower_red2 = np.array([170, 153, 102])  # Hue 170-180 (350-360 in 0-360 scale)
    upper_red2 = np.array([180, 255, 255])
    
    # Create mask for red regions
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Calculate minimum badge area based on screen size
    # At 1080p, badges are typically ~20px, so scale by total area
    total_pixels = h * w
    reference_pixels = 1920 * 1080  # Reference resolution
    scale_factor = (total_pixels / reference_pixels) ** 0.5
    min_area = int(20 * 20 * scale_factor)
    
    logger.debug(f"Badge detection: frame={w}x{h}, min_area={min_area}")
    
    for anchor_x, anchor_y in icon_anchors:
        # Define search region around anchor (±10% of screen)
        search_radius = 0.1
        
        # Convert anchor to pixels
        anchor_px_x = int(anchor_x * w)
        anchor_px_y = int(anchor_y * h)
        
        # Define search rectangle
        search_x1 = max(0, int((anchor_x - search_radius) * w))
        search_y1 = max(0, int((anchor_y - search_radius) * h))
        search_x2 = min(w, int((anchor_x + search_radius) * w))
        search_y2 = min(h, int((anchor_y + search_radius) * h))
        
        # Extract search region from mask
        search_mask = red_mask[search_y1:search_y2, search_x1:search_x2]
        
        if search_mask.size == 0:
            continue
        
        # Find contours in search region
        contours, _ = cv2.findContours(search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >= min_area:
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"]) + search_x1
                    cy = int(M["m01"] / M["m00"]) + search_y1
                    
                    # Convert back to normalized coordinates
                    badge_x = cx / w
                    badge_y = cy / h
                    
                    badges.append((badge_x, badge_y))
                    logger.debug(f"Red badge detected at ({badge_x:.3f}, {badge_y:.3f}) near anchor ({anchor_x:.3f}, {anchor_y:.3f})")
    
    logger.info(f"Detected {len(badges)} red badges")
    return badges


def identify_screen(frame: Frame, ocr_results: List[dict]) -> str:
    """Identify the current screen based on visible UI elements.
    
    Args:
        frame: Current screen frame
        ocr_results: OCR results from the frame
        
    Returns:
        Screen identifier string
    """
    # Extract all visible text
    visible_texts = [result['text'].lower() for result in ocr_results]
    
    # Define screen patterns
    screen_patterns = {
        'home': ['commissions', 'missions', 'mailbox', 'build', 'dock'],
        'commissions': ['commission', 'urgent', 'daily', 'weekly'],
        'mailbox': ['mail', 'collect', 'claim all', 'inbox'],
        'missions': ['mission', 'task', 'daily', 'weekly', 'event'],
        'build': ['build', 'construct', 'heavy', 'light', 'special'],
        'dock': ['dock', 'retire', 'enhance', 'limit break'],
        'shop': ['shop', 'purchase', 'buy', 'gem', 'coin'],
        'academy': ['academy', 'skill', 'training', 'lecture'],
        'exercise': ['exercise', 'pvp', 'arena', 'opponent'],
        'combat': ['sortie', 'battle', 'stage', 'chapter']
    }
    
    # Score each screen type
    screen_scores = {}
    for screen_name, keywords in screen_patterns.items():
        score = 0
        for keyword in keywords:
            for text in visible_texts:
                if keyword in text:
                    score += 1
                    break  # Only count each keyword once
        screen_scores[screen_name] = score
    
    # Find best match
    if screen_scores:
        best_screen = max(screen_scores.items(), key=lambda x: x[1])
        if best_screen[1] > 0:
            logger.debug(f"Identified screen: {best_screen[0]} (score: {best_screen[1]})")
            return best_screen[0]
    
    # Check for special UI patterns when OCR fails
    # Look for characteristic UI layouts
    
    # Home screen typically has bottom navigation
    if _has_bottom_navigation(frame):
        logger.debug("Identified screen: home (by UI layout)")
        return "home"
    
    # Default to unknown
    logger.debug("Could not identify screen")
    return "unknown"


def _has_bottom_navigation(frame: Frame) -> bool:
    """Check if frame has bottom navigation characteristic of home screen.
    
    Args:
        frame: Current screen frame
        
    Returns:
        True if bottom navigation is detected
    """
    # Extract bottom region
    h, w = frame.image_bgr.shape[:2]
    bottom_region = frame.image_bgr[int(h * 0.85):, :]
    
    if bottom_region.size == 0:
        return False
    
    # Convert to grayscale
    gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
    
    # Look for horizontal lines (navigation separators)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=w//4, maxLineGap=10)
    
    if lines is not None:
        # Count roughly horizontal lines
        horizontal_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 15 or angle > 165:  # Nearly horizontal
                horizontal_lines += 1
        
        # Home screen typically has at least one horizontal separator
        return horizontal_lines >= 1
    
    return False


def get_region_info() -> dict:
    """Get information about all defined regions.
    
    Returns:
        Dictionary mapping region names to their coordinates
    """
    return {
        'top_bar': Regions.top_bar,
        'bottom_bar': Regions.bottom_bar,
        'left_panel': Regions.left_panel,
        'center': Regions.center,
        'right_panel': Regions.right_panel
    }


def point_in_region(point: Tuple[float, float], region: Tuple[float, float, float, float]) -> bool:
    """Check if a point is within a region.
    
    Args:
        point: (x, y) normalized coordinates
        region: (x, y, w, h) normalized region
        
    Returns:
        True if point is within region
    """
    px, py = point
    rx, ry, rw, rh = region
    
    return rx <= px <= (rx + rw) and ry <= py <= (ry + rh)