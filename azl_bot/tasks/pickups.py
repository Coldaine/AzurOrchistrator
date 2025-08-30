"""Main menu pickups task."""

from typing import Any, Dict, List, Tuple

from loguru import logger

from ..core.capture import Frame
from ..core.screens import detect_red_badges, expected_home_elements, identify_screen


class PickupsTask:
    """Task to collect main menu pickups (mail, missions, etc.)."""
    
    name = "pickups"
    
    def goal(self) -> Dict[str, Any]:
        """Return goal description for LLM."""
        return {
            "action": "collect_pickups",
            "description": "Look for red notification badges on Mail, Missions, and other icons. Tap them to collect rewards, then return to Home."
        }
    
    def success(self, frame: Frame, context: Dict[str, Any]) -> bool:
        """Check if all pickups have been collected.
        
        Args:
            frame: Current frame
            context: Context including OCR results
            
        Returns:
            True if no more pickup badges are detected and we're on home screen
        """
        # Must be on home screen
        current_screen = context.get("last_screen", "unknown")
        if current_screen != "home":
            return False
        
        # Check for home screen elements
        if not self._is_home_screen(context.get("ocr_results", [])):
            return False
        
        # Check for red badges
        badges = self._detect_pickup_badges(frame)
        
        # Success if no badges found
        success = len(badges) == 0
        logger.info(f"Pickups task success check: {len(badges)} badges remaining")
        
        return success
    
    def on_success(self, planner, frame: Frame) -> None:
        """Handle successful pickup collection.
        
        Args:
            planner: Planner instance
            frame: Current frame
        """
        logger.info("All pickups collected successfully!")
        
        # Could record pickup statistics here if desired
        # For now, just log completion
    
    def _is_home_screen(self, ocr_results: List[dict]) -> bool:
        """Check if we're on the home screen based on OCR.
        
        Args:
            ocr_results: OCR results from current frame
            
        Returns:
            True if this appears to be the home screen
        """
        visible_texts = [result["text"].lower() for result in ocr_results]
        expected_elements = [elem.lower() for elem in expected_home_elements()]
        
        # Check if we can find at least 2 expected home elements
        found_count = 0
        for element in expected_elements:
            for text in visible_texts:
                if element in text:
                    found_count += 1
                    break
        
        return found_count >= 2
    
    def _detect_pickup_badges(self, frame: Frame) -> List[Tuple[float, float]]:
        """Detect red notification badges that indicate available pickups.
        
        Args:
            frame: Current frame
            
        Returns:
            List of (x, y) normalized coordinates where badges were detected
        """
        # Define common icon locations where badges might appear
        # These are typical positions for Mail, Missions, Build, etc.
        icon_anchors = [
            (0.1, 0.9),   # Bottom left area
            (0.3, 0.9),   # Bottom left-center
            (0.5, 0.9),   # Bottom center
            (0.7, 0.9),   # Bottom right-center
            (0.9, 0.9),   # Bottom right
            (0.95, 0.1),  # Top right (mail icon)
            (0.1, 0.5),   # Left side (missions)
            (0.9, 0.5),   # Right side (other menus)
        ]
        
        badges = detect_red_badges(frame, icon_anchors)
        logger.debug(f"Detected {len(badges)} pickup badges")
        
        return badges
    
    def get_next_pickup_target(self, frame: Frame, ocr_results: List[dict]) -> Dict[str, Any]:
        """Get the next pickup target to tap.
        
        Args:
            frame: Current frame
            ocr_results: OCR results
            
        Returns:
            Target description for LLM
        """
        # Detect badges
        badges = self._detect_pickup_badges(frame)
        
        if not badges:
            return {"action": "none", "reason": "no_badges_found"}
        
        # Get the first badge location
        badge_x, badge_y = badges[0]
        
        # Try to identify which icon this badge is near
        icon_name = self._identify_nearby_icon(badge_x, badge_y, ocr_results)
        
        if icon_name:
            return {
                "action": "tap_icon",
                "icon": icon_name,
                "location": [badge_x, badge_y],
                "reason": f"red_badge_near_{icon_name}"
            }
        else:
            return {
                "action": "tap_location", 
                "location": [badge_x, badge_y],
                "reason": "red_badge_unidentified"
            }
    
    def _identify_nearby_icon(self, badge_x: float, badge_y: float, ocr_results: List[dict]) -> str:
        """Try to identify which icon a badge is near.
        
        Args:
            badge_x: Badge X coordinate (normalized)
            badge_y: Badge Y coordinate (normalized)
            ocr_results: OCR results
            
        Returns:
            Icon name if identified, empty string otherwise
        """
        search_radius = 0.15  # 15% of screen
        
        # Look for text near the badge
        nearby_texts = []
        for result in ocr_results:
            box = result["box_norm"]
            text_center_x = box[0] + box[2] / 2
            text_center_y = box[1] + box[3] / 2
            
            # Calculate distance
            distance = ((badge_x - text_center_x)**2 + (badge_y - text_center_y)**2)**0.5
            
            if distance <= search_radius:
                nearby_texts.append(result["text"].lower())
        
        # Map text to icon names
        icon_mappings = {
            "mail": ["mail", "mailbox", "信箱", "メール"],
            "missions": ["mission", "task", "任務", "タスク"],
            "build": ["build", "construct", "建造", "建設"],
            "dock": ["dock", "fleet", "艦隊", "ドック"],
            "academy": ["academy", "school", "学院", "アカデミー"],
            "shop": ["shop", "store", "商店", "ショップ"],
            "commissions": ["commission", "委托", "委託"]
        }
        
        # Find best match
        for icon_name, keywords in icon_mappings.items():
            for text in nearby_texts:
                for keyword in keywords:
                    if keyword in text:
                        logger.debug(f"Identified badge near {icon_name} (matched '{keyword}' in '{text}')")
                        return icon_name
        
        return ""
    
    def should_attempt_claim(self, frame: Frame, ocr_results: List[dict]) -> bool:
        """Check if we should attempt to claim rewards on current screen.
        
        Args:
            frame: Current frame
            ocr_results: OCR results
            
        Returns:
            True if claim buttons are likely present
        """
        visible_texts = [result["text"].lower() for result in ocr_results]
        
        # Look for claim-related text
        claim_keywords = [
            "claim", "collect", "receive", "get", "take",
            "領取", "受取", "획득", "클레임"
        ]
        
        for text in visible_texts:
            for keyword in claim_keywords:
                if keyword in text:
                    logger.debug(f"Found claim keyword '{keyword}' in text '{text}'")
                    return True
        
        return False
    
    def get_claim_target(self, ocr_results: List[dict]) -> Dict[str, Any]:
        """Get target for claiming rewards.
        
        Args:
            ocr_results: OCR results
            
        Returns:
            Target description for claiming
        """
        # Look for claim buttons
        claim_texts = []
        
        for result in ocr_results:
            text = result["text"].lower()
            if any(keyword in text for keyword in ["claim", "collect", "receive", "領取", "受取"]):
                claim_texts.append({
                    "text": result["text"],
                    "box": result["box_norm"],
                    "conf": result["conf"]
                })
        
        if claim_texts:
            # Sort by confidence and take the best one
            best_claim = max(claim_texts, key=lambda x: x["conf"])
            
            return {
                "action": "tap_text",
                "text": best_claim["text"],
                "reason": "claim_button_found"
            }
        
        # Fallback: look for "claim all" type buttons
        for result in ocr_results:
            text = result["text"].lower()
            if "all" in text and any(keyword in text for keyword in ["claim", "collect", "領取"]):
                return {
                    "action": "tap_text", 
                    "text": result["text"],
                    "reason": "claim_all_button"
                }
        
        return {"action": "none", "reason": "no_claim_button"}


def create_pickups_task() -> PickupsTask:
    """Create and return a pickups task instance."""
    return PickupsTask()