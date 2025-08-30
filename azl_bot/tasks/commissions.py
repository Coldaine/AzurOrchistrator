"""Commissions reading task."""

import re
from typing import Any, Dict, List

from loguru import logger

from ..core.capture import Frame
from ..core.screens import Regions


class CommissionsTask:
    """Task to read commission information."""
    
    name = "commissions"
    
    def goal(self) -> Dict[str, Any]:
        """Return goal description for LLM."""
        return {
            "action": "read_commissions",
            "description": "Navigate to Commissions screen and read all visible commission slots with their status, names, rarity, and time remaining."
        }
    
    def success(self, frame: Frame, context: Dict[str, Any]) -> bool:
        """Check if commission reading was successful.
        
        Args:
            frame: Current frame
            context: Context including OCR results
            
        Returns:
            True if we're on commissions screen and extracted data
        """
        # Must be on commissions screen
        current_screen = context.get("last_screen", "unknown")
        if current_screen != "commissions":
            return False
        
        # Try to extract commission data
        commissions = self._extract_commissions(frame, context.get("ocr_results", []))
        
        # Success if we found at least 3 commission slots
        return len(commissions) >= 3
    
    def on_success(self, planner, frame: Frame) -> None:
        """Handle successful commission reading.
        
        Args:
            planner: Planner instance
            frame: Current frame
        """
        # Run OCR again to get fresh results
        ocr_results = planner.ocr.text_in_roi(frame.image_bgr, (0, 0, 1, 1))
        
        # Extract commission data
        commissions = self._extract_commissions(frame, ocr_results)
        
        # Store in database
        planner.datastore.record_commissions(commissions)
        
        logger.info(f"Recorded {len(commissions)} commission entries")
    
    def _extract_commissions(self, frame: Frame, ocr_results: List[dict]) -> List[Dict[str, Any]]:
        """Extract commission data from the screen.
        
        Args:
            frame: Current frame
            ocr_results: OCR results from full frame
            
        Returns:
            List of commission dictionaries
        """
        commissions = []
        
        # Filter OCR results to center region where commission list typically appears
        center_results = self._filter_ocr_to_region(ocr_results, Regions.center)
        
        # Group OCR results by vertical position (rows)
        commission_rows = self._group_into_rows(center_results)
        
        for slot_id, row_results in enumerate(commission_rows):
            commission = self._parse_commission_row(slot_id, row_results)
            if commission:
                commissions.append(commission)
        
        logger.info(f"Extracted {len(commissions)} commissions from screen")
        return commissions
    
    def _filter_ocr_to_region(self, ocr_results: List[dict], region: tuple) -> List[dict]:
        """Filter OCR results to those within a specific region.
        
        Args:
            ocr_results: Full OCR results
            region: (x, y, w, h) normalized region
            
        Returns:
            Filtered OCR results
        """
        rx, ry, rw, rh = region
        filtered = []
        
        for result in ocr_results:
            box = result["box_norm"]
            box_center_x = box[0] + box[2] / 2
            box_center_y = box[1] + box[3] / 2
            
            # Check if center point is within region
            if (rx <= box_center_x <= (rx + rw) and 
                ry <= box_center_y <= (ry + rh)):
                filtered.append(result)
        
        return filtered
    
    def _group_into_rows(self, ocr_results: List[dict], row_threshold: float = 0.05) -> List[List[dict]]:
        """Group OCR results into rows based on Y position.
        
        Args:
            ocr_results: OCR results to group
            row_threshold: Maximum Y difference to consider same row
            
        Returns:
            List of rows, each containing OCR results
        """
        if not ocr_results:
            return []
        
        # Sort by Y position
        sorted_results = sorted(ocr_results, key=lambda r: r["box_norm"][1])
        
        rows = []
        current_row = [sorted_results[0]]
        current_y = sorted_results[0]["box_norm"][1]
        
        for result in sorted_results[1:]:
            result_y = result["box_norm"][1]
            
            if abs(result_y - current_y) <= row_threshold:
                # Same row
                current_row.append(result)
            else:
                # New row
                rows.append(current_row)
                current_row = [result]
                current_y = result_y
        
        # Add the last row
        if current_row:
            rows.append(current_row)
        
        return rows
    
    def _parse_commission_row(self, slot_id: int, row_results: List[dict]) -> Dict[str, Any]:
        """Parse a single commission row.
        
        Args:
            slot_id: Commission slot identifier
            row_results: OCR results from this row
            
        Returns:
            Commission dictionary or None if parsing failed
        """
        if not row_results:
            return None
        
        # Sort row results by X position (left to right)
        sorted_row = sorted(row_results, key=lambda r: r["box_norm"][0])
        
        # Extract text from the row
        row_texts = [result["text"].strip() for result in sorted_row]
        combined_text = " ".join(row_texts)
        
        logger.debug(f"Parsing commission row {slot_id}: {combined_text}")
        
        commission = {
            "slot_id": slot_id,
            "name": "",
            "rarity": "",
            "time_remaining_s": None,
            "status": "unknown"
        }
        
        # Extract time remaining
        time_remaining = self._extract_time_remaining(row_texts)
        if time_remaining is not None:
            commission["time_remaining_s"] = time_remaining
        
        # Determine status based on content
        commission["status"] = self._determine_commission_status(row_texts, time_remaining)
        
        # Extract commission name (usually the longest non-time text)
        commission["name"] = self._extract_commission_name(row_texts)
        
        # Extract rarity if possible
        commission["rarity"] = self._extract_commission_rarity(row_texts)
        
        return commission
    
    def _extract_time_remaining(self, row_texts: List[str]) -> int:
        """Extract time remaining from row texts.
        
        Args:
            row_texts: List of text strings from the row
            
        Returns:
            Time remaining in seconds, or None if not found
        """
        time_patterns = [
            r'(\d{1,2}):(\d{2}):(\d{2})',  # HH:MM:SS
            r'(\d{1,2}):(\d{2})',          # MM:SS or HH:MM
            r'(\d+)h\s*(\d+)m',            # 1h 30m
            r'(\d+)m\s*(\d+)s',            # 30m 45s
            r'(\d+)\s*hours?',             # 5 hours
            r'(\d+)\s*mins?',              # 30 minutes
        ]
        
        for text in row_texts:
            for pattern in time_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return self._parse_time_match(match, pattern)
        
        return None
    
    def _parse_time_match(self, match, pattern: str) -> int:
        """Parse time match into seconds.
        
        Args:
            match: Regex match object
            pattern: The pattern that matched
            
        Returns:
            Time in seconds
        """
        groups = match.groups()
        
        if ':' in pattern:
            if len(groups) == 3:  # HH:MM:SS
                hours, minutes, seconds = map(int, groups)
                return hours * 3600 + minutes * 60 + seconds
            elif len(groups) == 2:  # MM:SS or HH:MM
                # Assume MM:SS if first number < 60, otherwise HH:MM
                first, second = map(int, groups)
                if first < 60:  # MM:SS
                    return first * 60 + second
                else:  # HH:MM
                    return first * 3600 + second * 60
        elif 'h' in pattern and 'm' in pattern:  # 1h 30m
            hours, minutes = map(int, groups)
            return hours * 3600 + minutes * 60
        elif 'm' in pattern and 's' in pattern:  # 30m 45s
            minutes, seconds = map(int, groups)
            return minutes * 60 + seconds
        elif 'hour' in pattern:  # 5 hours
            hours = int(groups[0])
            return hours * 3600
        elif 'min' in pattern:  # 30 minutes
            minutes = int(groups[0])
            return minutes * 60
        
        return 0
    
    def _determine_commission_status(self, row_texts: List[str], time_remaining: int) -> str:
        """Determine commission status.
        
        Args:
            row_texts: Text from the row
            time_remaining: Time remaining in seconds
            
        Returns:
            Status string: "idle", "in_progress", "ready"
        """
        combined_text = " ".join(row_texts).lower()
        
        # Check for explicit status indicators
        if any(keyword in combined_text for keyword in ["complete", "ready", "finished", "完成", "完了"]):
            return "ready"
        
        if any(keyword in combined_text for keyword in ["in progress", "running", "進行中", "実行中"]):
            return "in_progress"
        
        if any(keyword in combined_text for keyword in ["idle", "empty", "available", "空闲", "待機"]):
            return "idle"
        
        # Infer from time remaining
        if time_remaining is not None:
            if time_remaining > 0:
                return "in_progress"
            else:
                return "ready"
        
        # Default to idle if we can't determine
        return "idle"
    
    def _extract_commission_name(self, row_texts: List[str]) -> str:
        """Extract commission name from row texts.
        
        Args:
            row_texts: Text from the row
            
        Returns:
            Commission name or empty string
        """
        # Filter out time strings and status words
        filtered_texts = []
        
        for text in row_texts:
            # Skip if it looks like a time
            if re.search(r'\d+:\d+', text) or re.search(r'\d+[hms]', text, re.IGNORECASE):
                continue
            
            # Skip status words
            if any(word in text.lower() for word in ["complete", "ready", "idle", "progress", "完成", "進行", "待機"]):
                continue
            
            # Skip very short texts (likely UI elements)
            if len(text.strip()) < 3:
                continue
            
            filtered_texts.append(text.strip())
        
        # Return the longest remaining text as the name
        if filtered_texts:
            return max(filtered_texts, key=len)
        
        return ""
    
    def _extract_commission_rarity(self, row_texts: List[str]) -> str:
        """Extract commission rarity if available.
        
        Args:
            row_texts: Text from the row
            
        Returns:
            Rarity string or empty string
        """
        combined_text = " ".join(row_texts).lower()
        
        # Common rarity indicators
        rarity_keywords = {
            "urgent": ["urgent", "緊急", "緊急"],
            "daily": ["daily", "每日", "デイリー"],
            "weekly": ["weekly", "每週", "ウィークリー"],
            "normal": ["normal", "一般", "通常"],
            "major": ["major", "重大", "メジャー"]
        }
        
        for rarity, keywords in rarity_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                return rarity
        
        return ""


def create_commissions_task() -> CommissionsTask:
    """Create and return a commissions task instance."""
    return CommissionsTask()