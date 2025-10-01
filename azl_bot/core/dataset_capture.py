"""Dataset capture for creating reproducible test sets."""

import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np
from loguru import logger
from PIL import Image


def compute_dhash(image: np.ndarray, hash_size: int = 8) -> str:
    """Compute difference hash (dhash) of an image.
    
    Args:
        image: Image array (BGR or grayscale)
        hash_size: Size of hash (default 8 for 64-bit hash)
        
    Returns:
        Hexadecimal hash string
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize to hash_size + 1 to compute differences
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    
    # Compute horizontal gradient
    diff = resized[:, 1:] > resized[:, :-1]
    
    # Convert to hash string
    hash_int = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    return format(hash_int, 'x').zfill(hash_size * hash_size // 4)


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hash strings.
    
    Args:
        hash1: First hash string
        hash2: Second hash string
        
    Returns:
        Hamming distance (number of differing bits)
    """
    if len(hash1) != len(hash2):
        return max(len(hash1), len(hash2)) * 4
    
    distance = 0
    for c1, c2 in zip(hash1, hash2):
        xor = int(c1, 16) ^ int(c2, 16)
        distance += bin(xor).count('1')
    
    return distance


class DatasetCapture:
    """Manages dataset capture with deduplication and retention."""
    
    def __init__(self, config: Dict[str, Any], base_dir: Path):
        """Initialize dataset capture.
        
        Args:
            config: Capture configuration from DataCaptureConfig
            base_dir: Base directory for dataset storage
        """
        self.config = config
        self.base_dir = base_dir / "dataset"
        self.enabled = config.get("enabled", False)
        self.sample_rate_hz = config.get("sample_rate_hz", 0.5)
        self.max_dim = config.get("max_dim", 1280)
        self.format = config.get("format", "jpg")
        self.jpeg_quality = config.get("jpeg_quality", 85)
        
        dedupe_config = config.get("dedupe", {})
        self.dedupe_method = dedupe_config.get("method", "dhash")
        self.hamming_threshold = dedupe_config.get("hamming_threshold", 3)
        
        retention_config = config.get("retention", {})
        self.max_files = retention_config.get("max_files", 2000)
        self.max_days = retention_config.get("max_days", 60)
        
        self.save_metadata = config.get("metadata", True)
        
        # State
        self._last_capture_time = 0.0
        self._recent_hashes = []  # Keep recent hashes for dedup
        self._session_count = 0
        
        # Create base directory
        if self.enabled:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Dataset capture initialized at {self.base_dir}")
    
    def should_capture(self) -> bool:
        """Check if enough time has passed for next capture.
        
        Returns:
            True if should capture based on sample rate
        """
        if not self.enabled:
            return False
        
        now = time.time()
        interval = 1.0 / self.sample_rate_hz if self.sample_rate_hz > 0 else float('inf')
        
        if now - self._last_capture_time >= interval:
            return True
        
        return False
    
    def capture(
        self, 
        image_bgr: np.ndarray, 
        active_rect: tuple, 
        full_size: tuple,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Path]:
        """Capture an image to dataset if not duplicate.
        
        Args:
            image_bgr: Image to capture (BGR format, active area only)
            active_rect: Active rectangle (x, y, w, h)
            full_size: Full frame size (width, height)
            context: Optional context dict (screen, task, etc.)
            
        Returns:
            Path to saved file, or None if skipped
        """
        if not self.should_capture():
            return None
        
        self._last_capture_time = time.time()
        
        # Compute hash for dedup
        img_hash = compute_dhash(image_bgr)
        
        # Check for duplicates
        is_duplicate = False
        for recent_hash in self._recent_hashes[-20:]:  # Check last 20 hashes
            if hamming_distance(img_hash, recent_hash) <= self.hamming_threshold:
                is_duplicate = True
                break
        
        if is_duplicate:
            logger.debug("Skipping duplicate image")
            return None
        
        # Add to recent hashes
        self._recent_hashes.append(img_hash)
        if len(self._recent_hashes) > 100:
            self._recent_hashes = self._recent_hashes[-50:]
        
        # Resize if needed
        h, w = image_bgr.shape[:2]
        if max(w, h) > self.max_dim:
            scale = self.max_dim / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image_resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            image_resized = image_bgr
        
        # Get today's directory
        today = datetime.now().strftime("%Y%m%d")
        day_dir = self.base_dir / today
        day_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision
        short_hash = img_hash[:8]
        
        if self.format == "jpg":
            filename = f"{timestamp}_{short_hash}.jpg"
            image_path = day_dir / filename
            
            # Save as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            cv2.imwrite(str(image_path), image_resized, encode_params)
        else:
            filename = f"{timestamp}_{short_hash}.png"
            image_path = day_dir / filename
            cv2.imwrite(str(image_path), image_resized)
        
        # Save metadata if requested
        if self.save_metadata:
            metadata = {
                "timestamp": time.time(),
                "timestamp_str": datetime.now().isoformat(),
                "hash": img_hash,
                "frame_size": {
                    "full_width": full_size[0],
                    "full_height": full_size[1]
                },
                "active_rect": {
                    "x": active_rect[0],
                    "y": active_rect[1],
                    "width": active_rect[2],
                    "height": active_rect[3]
                },
                "captured_size": {
                    "width": image_resized.shape[1],
                    "height": image_resized.shape[0]
                }
            }
            
            if context:
                metadata["context"] = context
            
            metadata_path = day_dir / f"{timestamp}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        self._session_count += 1
        logger.debug(f"Captured dataset image: {filename}")
        
        # Run retention cleanup periodically
        if self._session_count % 10 == 0:
            self._cleanup_old_files()
        
        return image_path
    
    def _cleanup_old_files(self):
        """Clean up old files based on retention policy."""
        if not self.base_dir.exists():
            return
        
        # Gather all captured files with timestamps
        all_files = []
        for day_dir in self.base_dir.iterdir():
            if not day_dir.is_dir():
                continue
            
            for file_path in day_dir.glob("*.[jp][pn]g"):
                try:
                    mtime = file_path.stat().st_mtime
                    all_files.append((mtime, file_path))
                except Exception:
                    pass
        
        # Sort by modification time (oldest first)
        all_files.sort()
        
        # Remove by age
        cutoff_time = time.time() - (self.max_days * 86400)
        files_to_remove = []
        
        for mtime, file_path in all_files:
            if mtime < cutoff_time:
                files_to_remove.append(file_path)
        
        # Remove by count
        if len(all_files) > self.max_files:
            excess_count = len(all_files) - self.max_files
            for _, file_path in all_files[:excess_count]:
                if file_path not in files_to_remove:
                    files_to_remove.append(file_path)
        
        # Perform removal
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                # Also remove metadata if exists
                metadata_path = file_path.parent / f"{file_path.stem.split('_')[0]}_{file_path.stem.split('_')[1]}.json"
                if metadata_path.exists():
                    metadata_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
        
        if files_to_remove:
            logger.info(f"Cleaned up {len(files_to_remove)} old dataset files")
        
        # Remove empty directories
        for day_dir in self.base_dir.iterdir():
            if day_dir.is_dir() and not any(day_dir.iterdir()):
                try:
                    day_dir.rmdir()
                except Exception:
                    pass
    
    def get_session_count(self) -> int:
        """Get number of images captured this session.
        
        Returns:
            Count of images captured
        """
        return self._session_count
    
    def get_current_day_dir(self) -> Path:
        """Get current day's capture directory.
        
        Returns:
            Path to today's directory
        """
        today = datetime.now().strftime("%Y%m%d")
        return self.base_dir / today
    
    def toggle_enabled(self) -> bool:
        """Toggle enabled state.
        
        Returns:
            New enabled state
        """
        self.enabled = not self.enabled
        logger.info(f"Dataset capture {'enabled' if self.enabled else 'disabled'}")
        return self.enabled
    
    def set_enabled(self, enabled: bool):
        """Set enabled state.
        
        Args:
            enabled: New enabled state
        """
        self.enabled = enabled
        logger.info(f"Dataset capture {'enabled' if self.enabled else 'disabled'}")
