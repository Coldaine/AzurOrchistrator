"""Enhanced logging and frame storage."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from loguru import logger

from .capture import Frame
from .configs import LoggingConfig


class RunLogger:
    """Logger for individual task runs with frame and action storage."""
    
    def __init__(self, run_id: int, base_dir: Path, config: LoggingConfig) -> None:
        """Initialize run logger.
        
        Args:
            run_id: Unique run identifier
            base_dir: Base directory for log storage
            config: Logging configuration
        """
        self.run_id = run_id
        self.config = config
        
        # Create run directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = base_dir / "runs" / f"{timestamp}_{run_id:04d}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.frames_dir = self.run_dir / "frames"
        self.overlays_dir = self.run_dir / "overlays"
        
        if config.keep_frames:
            self.frames_dir.mkdir(exist_ok=True)
        
        if config.overlay_draw:
            self.overlays_dir.mkdir(exist_ok=True)
        
        # Initialize JSONL log file
        self.actions_file = self.run_dir / "actions.jsonl"
        
        logger.info(f"Run logger initialized: {self.run_dir}")
    
    def log_action(self, screen: Optional[str], action: str, selector_json: Optional[str] = None,
                  method: Optional[str] = None, point_norm_x: Optional[float] = None, 
                  point_norm_y: Optional[float] = None, confidence: Optional[float] = None,
                  success: Optional[bool] = None) -> None:
        """Log an action to JSONL file.
        
        Args:
            screen: Screen identifier
            action: Action type
            selector_json: JSON representation of selector
            method: Resolution method used
            point_norm_x: Normalized X coordinate
            point_norm_y: Normalized Y coordinate
            confidence: Confidence score
            success: Whether action was successful
        """
        action_data = {
            "ts": time.time(),
            "screen": screen,
            "action": action,
            "selector_json": selector_json,
            "method": method,
            "point_norm_x": point_norm_x,
            "point_norm_y": point_norm_y,
            "confidence": confidence,
            "success": success
        }
        
        # Write to JSONL file
        with open(self.actions_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(action_data) + '\n')
    
    def save_frame(self, frame: Frame, frame_id: Optional[str] = None, with_overlay: bool = False,
                  overlay_data: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """Save frame to disk.
        
        Args:
            frame: Frame to save
            frame_id: Optional frame identifier (uses timestamp if None)
            with_overlay: Whether to draw overlay annotations
            overlay_data: Data for overlay drawing
            
        Returns:
            Path to saved frame file, or None if saving disabled
        """
        if not self.config.keep_frames:
            return None
        
        if frame_id is None:
            frame_id = f"frame_{int(frame.ts * 1000):013d}"
        
        # Save original frame
        frame_path = self.frames_dir / f"{frame_id}.png"
        
        # Convert back to RGB for saving
        frame_rgb = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(frame_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        
        # Save overlay if requested
        if with_overlay and self.config.overlay_draw and overlay_data:
            overlay_path = self._save_overlay(frame, frame_id, overlay_data)
            logger.debug(f"Saved frame with overlay: {overlay_path}")
        
        logger.debug(f"Saved frame: {frame_path}")
        return frame_path
    
    def _save_overlay(self, frame: Frame, frame_id: str, overlay_data: Dict[str, Any]) -> Optional[Path]:
        """Save frame with overlay annotations.
        
        Args:
            frame: Original frame
            frame_id: Frame identifier
            overlay_data: Overlay drawing data
            
        Returns:
            Path to overlay file
        """
        overlay_path = self.overlays_dir / f"{frame_id}_overlay.png"
        
        # Start with original frame
        overlay_img = frame.image_bgr.copy()
        h, w = overlay_img.shape[:2]
        
        # Draw regions if provided
        if "regions" in overlay_data:
            self._draw_regions(overlay_img, overlay_data["regions"])
        
        # Draw candidate points if provided
        if "candidates" in overlay_data:
            self._draw_candidates(overlay_img, overlay_data["candidates"])
        
        # Draw last tap point if provided
        if "last_tap" in overlay_data:
            self._draw_tap_point(overlay_img, overlay_data["last_tap"])
        
        # Draw bounding boxes if provided
        if "boxes" in overlay_data:
            self._draw_boxes(overlay_img, overlay_data["boxes"])
        
        # Save overlay image
        cv2.imwrite(str(overlay_path), overlay_img)
        return overlay_path
    
    def _draw_regions(self, img: np.ndarray, regions: Dict[str, tuple]) -> None:
        """Draw region rectangles on image.
        
        Args:
            img: Image to draw on
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
            cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 255), 2)
            
            # Draw label
            cv2.putText(img, region_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
    
    def _draw_candidates(self, img: np.ndarray, candidates: list) -> None:
        """Draw candidate points on image.
        
        Args:
            img: Image to draw on
            candidates: List of candidate data
        """
        h, w = img.shape[:2]
        
        for i, candidate in enumerate(candidates):
            if "point" in candidate and "confidence" in candidate:
                x, y = candidate["point"]
                conf = candidate["confidence"]
                method = candidate.get("method", "unknown")
                
                # Convert to pixels
                px = int(x * w)
                py = int(y * h)
                
                # Choose color based on confidence
                if conf >= 0.8:
                    color = (0, 255, 0)  # Green
                elif conf >= 0.6:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                # Draw circle
                cv2.circle(img, (px, py), 8, color, 2)
                
                # Draw label
                label = f"{method}:{conf:.2f}"
                cv2.putText(img, label, (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _draw_tap_point(self, img: np.ndarray, tap_data: dict) -> None:
        """Draw last tap point on image.
        
        Args:
            img: Image to draw on
            tap_data: Tap point data
        """
        if "point" not in tap_data:
            return
        
        h, w = img.shape[:2]
        x, y = tap_data["point"]
        
        # Convert to pixels
        px = int(x * w)
        py = int(y * h)
        
        # Draw large circle for tap
        cv2.circle(img, (px, py), 15, (255, 0, 255), 3)  # Magenta
        cv2.circle(img, (px, py), 3, (255, 255, 255), -1)  # White center
        
        # Draw timestamp if available
        if "timestamp" in tap_data:
            ts = tap_data["timestamp"]
            label = f"TAP @{ts:.1f}s"
            cv2.putText(img, label, (px + 20, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    def _draw_boxes(self, img: np.ndarray, boxes: list) -> None:
        """Draw bounding boxes on image.
        
        Args:
            img: Image to draw on
            boxes: List of box data
        """
        h, w = img.shape[:2]
        
        for box_data in boxes:
            if "bbox" in box_data:
                bx, by, bw, bh = box_data["bbox"]
                
                # Convert to pixels
                x1 = int(bx * w)
                y1 = int(by * h)
                x2 = int((bx + bw) * w)
                y2 = int((by + bh) * h)
                
                # Draw rectangle
                color = (255, 255, 0)  # Cyan
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label if available
                if "label" in box_data:
                    label = box_data["label"]
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def get_run_stats(self) -> Dict[str, Any]:
        """Get statistics for this run.
        
        Returns:
            Dictionary with run statistics
        """
        stats = {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "actions_count": 0,
            "frames_count": 0,
            "overlays_count": 0
        }
        
        # Count actions
        if self.actions_file.exists():
            with open(self.actions_file, 'r', encoding='utf-8') as f:
                stats["actions_count"] = sum(1 for _ in f)
        
        # Count frames
        if self.frames_dir.exists():
            stats["frames_count"] = len(list(self.frames_dir.glob("*.png")))
        
        # Count overlays
        if self.overlays_dir.exists():
            stats["overlays_count"] = len(list(self.overlays_dir.glob("*.png")))
        
        return stats


def init_logger(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Initialize global logger configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with colors
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            sink=str(log_file),
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="30 days"
        )
    
    logger.info(f"Logger initialized with level: {level}")


def open_run_dir(run_id: int, base_dir: Path, config: LoggingConfig) -> RunLogger:
    """Open a new run logger.
    
    Args:
        run_id: Run identifier
        base_dir: Base directory for logs
        config: Logging configuration
        
    Returns:
        RunLogger instance
    """
    return RunLogger(run_id, base_dir, config)


class TelemetryTracker:
    """Structured telemetry tracking for loop metrics."""
    
    def __init__(self):
        """Initialize telemetry tracker."""
        self.counters: Dict[str, int] = {}
        self.timings: Dict[str, list[float]] = {}
        self._start_times: Dict[str, float] = {}
    
    def increment(self, counter_name: str, amount: int = 1) -> None:
        """Increment a counter.
        
        Args:
            counter_name: Name of counter to increment
            amount: Amount to increment by (default 1)
        """
        self.counters[counter_name] = self.counters.get(counter_name, 0) + amount
        logger.debug(f"Counter '{counter_name}' incremented to {self.counters[counter_name]}")
    
    def start_timing(self, operation: str) -> None:
        """Start timing an operation.
        
        Args:
            operation: Name of operation being timed
        """
        self._start_times[operation] = time.time()
    
    def end_timing(self, operation: str) -> Optional[float]:
        """End timing an operation and record duration.
        
        Args:
            operation: Name of operation being timed
            
        Returns:
            Duration in seconds, or None if timing was not started
        """
        if operation not in self._start_times:
            logger.warning(f"Timing '{operation}' was not started")
            return None
        
        duration = time.time() - self._start_times.pop(operation)
        
        if operation not in self.timings:
            self.timings[operation] = []
        self.timings[operation].append(duration)
        
        logger.debug(f"Operation '{operation}' took {duration:.3f}s")
        return duration
    
    def get_average_timing(self, operation: str) -> Optional[float]:
        """Get average timing for an operation.
        
        Args:
            operation: Name of operation
            
        Returns:
            Average duration in seconds, or None if no timings recorded
        """
        if operation not in self.timings or not self.timings[operation]:
            return None
        return sum(self.timings[operation]) / len(self.timings[operation])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all telemetry statistics.
        
        Returns:
            Dictionary with counters and timing averages
        """
        stats = {"counters": dict(self.counters)}
        
        timing_avgs = {}
        for operation, durations in self.timings.items():
            if durations:
                timing_avgs[operation] = {
                    "avg": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "count": len(durations)
                }
        
        if timing_avgs:
            stats["timings"] = timing_avgs
        
        return stats
    
    def log_stats(self) -> None:
        """Log current statistics."""
        stats = self.get_stats()
        logger.info(f"Telemetry stats: {json.dumps(stats, indent=2)}")
    
    def reset(self) -> None:
        """Reset all counters and timings."""
        self.counters.clear()
        self.timings.clear()
        self._start_times.clear()
        logger.debug("Telemetry tracker reset")