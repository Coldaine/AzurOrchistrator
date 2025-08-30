"""UI state management."""

import time
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class TapPoint:
    """Information about a tap point."""
    x: float
    y: float
    timestamp: float
    method: str = "unknown"
    confidence: float = 1.0


class UIState:
    """Manages UI state and shared data."""
    
    def __init__(self) -> None:
        """Initialize UI state."""
        self.last_tap_point: Optional[Dict[str, Any]] = None
        self.current_screen = "unknown"
        self.last_frame_time = 0.0
        self.last_plan_json = ""
        self.current_run_id: Optional[int] = None
        self.task_running = False
        self.last_ocr_results = []
        self.last_candidates = []
        
    def record_tap(self, x: float, y: float, method: str = "manual", confidence: float = 1.0) -> None:
        """Record a tap point.
        
        Args:
            x: Normalized X coordinate
            y: Normalized Y coordinate
            method: Method used for tap
            confidence: Confidence score
        """
        self.last_tap_point = {
            "point": (x, y),
            "timestamp": time.time(),
            "method": method,
            "confidence": confidence
        }
    
    def update_screen(self, screen: str) -> None:
        """Update current screen.
        
        Args:
            screen: Screen identifier
        """
        self.current_screen = screen
    
    def update_frame_time(self, timestamp: float) -> None:
        """Update last frame timestamp.
        
        Args:
            timestamp: Frame timestamp
        """
        self.last_frame_time = timestamp
    
    def update_plan_json(self, json_str: str) -> None:
        """Update last LLM plan JSON.
        
        Args:
            json_str: JSON string from LLM
        """
        self.last_plan_json = json_str
    
    def set_run_id(self, run_id: Optional[int]) -> None:
        """Set current run ID.
        
        Args:
            run_id: Run identifier or None if no run active
        """
        self.current_run_id = run_id
        self.task_running = run_id is not None
    
    def update_ocr_results(self, results: list) -> None:
        """Update OCR results.
        
        Args:
            results: List of OCR results
        """
        self.last_ocr_results = results
    
    def update_candidates(self, candidates: list) -> None:
        """Update candidate points.
        
        Args:
            candidates: List of candidate points
        """
        self.last_candidates = candidates
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get current status information.
        
        Returns:
            Dictionary with status information
        """
        return {
            "screen": self.current_screen,
            "frame_time": self.last_frame_time,
            "run_id": self.current_run_id,
            "task_running": self.task_running,
            "last_tap": self.last_tap_point,
            "ocr_count": len(self.last_ocr_results),
            "candidates_count": len(self.last_candidates)
        }
    
    def clear_transient_data(self) -> None:
        """Clear transient data (candidates, OCR results, etc.)."""
        self.last_candidates = []
        self.last_ocr_results = []
        
    def is_tap_recent(self, max_age: float = 5.0) -> bool:
        """Check if last tap is recent.
        
        Args:
            max_age: Maximum age in seconds
            
        Returns:
            True if last tap is within max_age
        """
        if not self.last_tap_point:
            return False
        
        age = time.time() - self.last_tap_point["timestamp"]
        return age <= max_age