"""Image hashing for efficient frame deduplication."""

import cv2
import numpy as np
from typing import Optional
import imagehash
from PIL import Image

class FrameHasher:
    """Detects meaningful frame changes using perceptual hashing."""
    
    def __init__(self, hash_size: int = 16, similarity_threshold: float = 0.95):
        self.hash_size = hash_size
        self.similarity_threshold = similarity_threshold
        self._last_hash: Optional[imagehash.ImageHash] = None
        self._last_frame_data: Optional[np.ndarray] = None
        self._stable_count: int = 0
        self._stability_history: list[imagehash.ImageHash] = []
    
    def compute_hash(self, image: np.ndarray) -> imagehash.ImageHash:
        """Compute perceptual hash of image."""
        # Convert BGR to RGB PIL Image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Use difference hash for speed
        return imagehash.dhash(pil_image, hash_size=self.hash_size)
    
    def has_changed(self, image: np.ndarray) -> bool:
        """Check if frame has meaningfully changed."""
        current_hash = self.compute_hash(image)
        
        if self._last_hash is None:
            self._last_hash = current_hash
            return True
        
        # Calculate similarity (0-1, where 1 is identical)
        hash_diff = 1 - (current_hash - self._last_hash) / (self.hash_size * self.hash_size * 4)
        
        if hash_diff < self.similarity_threshold:
            self._last_hash = current_hash
            return True
        
        return False
    
    def is_stable(self, image: np.ndarray, required_matches: int = 3) -> bool:
        """Check if frame has been stable for required number of frames.
        
        Args:
            image: Image to check
            required_matches: Number of consecutive matching frames needed for stability
            
        Returns:
            True if frame has been stable for required_matches frames, False otherwise
        """
        current_hash = self.compute_hash(image)
        
        # Add current hash to history
        self._stability_history.append(current_hash)
        
        # Keep only the most recent hashes (required_matches)
        if len(self._stability_history) > required_matches:
            self._stability_history.pop(0)
        
        # Check if we have enough history
        if len(self._stability_history) < required_matches:
            return False
        
        # Check if all hashes in history match within threshold
        reference_hash = self._stability_history[0]
        for h in self._stability_history[1:]:
            # Calculate hamming distance
            distance = reference_hash - h
            # If any hash is too different, not stable
            if distance > (self.hash_size * self.hash_size * (1 - self.similarity_threshold)):
                # Reset history when instability detected
                self._stability_history = [current_hash]
                return False
        
        # All frames match - we're stable
        return True
    
    def reset(self):
        """Reset hash tracking."""
        self._last_hash = None
        self._last_frame_data = None
        self._stable_count = 0
        self._stability_history = []