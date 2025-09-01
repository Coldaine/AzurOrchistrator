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
    
    def reset(self):
        """Reset hash tracking."""
        self._last_hash = None
        self._last_frame_data = None