"""Image hashing for efficient frame deduplication (no external deps)."""

from typing import Optional

import cv2
import numpy as np


def _dhash_int(image: np.ndarray, hash_size: int) -> int:
    """Compute a difference hash and return it as an integer.

    Steps:
    1) convert to grayscale
    2) resize to (hash_size+1, hash_size)
    3) compare adjacent pixels horizontally -> bits
    4) pack bits row-major into an int
    """

    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Ensure uint8
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diffs = resized[:, 1:] > resized[:, :-1]
    # Flatten row-major
    bits = diffs.flatten()

    # Pack into integer
    h = 0
    for bit in bits:
        h = (h << 1) | int(bit)
    return h


def _hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class FrameHasher:
    """Detects meaningful frame changes using perceptual hashing.

    Attributes:
        hash_size: Size of one dimension of the dHash (nbits = hash_size*hash_size)
        similarity_threshold: If similarity < threshold, considered changed.
        extra_intensity_bits: Extra MSBs from mean intensity appended to reduce
            collisions on uniform frames (0-8).
    """

    def __init__(self, hash_size: int = 16, similarity_threshold: float = 0.95, extra_intensity_bits: int = 8):
        self.hash_size = int(hash_size)
        self.similarity_threshold = float(similarity_threshold)
        self.extra_intensity_bits = int(max(0, min(8, extra_intensity_bits)))
        self._nbits = self.hash_size * self.hash_size + self.extra_intensity_bits
        self._last_hash: Optional[int] = None

        # Stability tracking
        self._baseline_hash: Optional[int] = None
        self._stable_count: int = 0

    def compute_hash(self, image: np.ndarray) -> int:
        """Compute perceptual difference hash as an integer."""
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        h = _dhash_int(gray, self.hash_size)
        if self.extra_intensity_bits:
            # Use float mean to satisfy type checkers and convert to int range 0..255
            mean_val = int(float(np.mean(gray.astype(np.float32, copy=False))))  # 0..255
            if self.extra_intensity_bits < 8:
                # Keep the most significant bits to preserve ordering
                mean_val = mean_val >> (8 - self.extra_intensity_bits)
            # Mask to the desired width
            mean_val &= (1 << self.extra_intensity_bits) - 1
            h = (h << self.extra_intensity_bits) | mean_val
        return h

    def _similarity(self, h1: int, h2: int) -> float:
        nbits = self._nbits
        dist = _hamming_distance(h1, h2)
        return 1.0 - (dist / nbits)

    def has_changed(self, image: np.ndarray) -> bool:
        """Check if frame has meaningfully changed since last seen.

        Returns True on first call (no reference) or when similarity drops
        below the configured threshold. Updates the last-hash reference when
        a change is detected or when first initialized.
        """
        current_hash = self.compute_hash(image)

        if self._last_hash is None:
            self._last_hash = current_hash
            return True

        sim = self._similarity(current_hash, self._last_hash)
        if sim < self.similarity_threshold:
            self._last_hash = current_hash
            return True
        return False

    def is_stable(self, image: np.ndarray, required_matches: int = 3) -> bool:
        """Return True once the same image is observed N times (within threshold).

        The first call establishes a baseline and returns False.
        Subsequent calls increment a stability counter while the similarity
        vs. baseline is >= threshold; a change resets the baseline and counter.
        """
        current_hash = self.compute_hash(image)

        if self._baseline_hash is None:
            self._baseline_hash = current_hash
            self._stable_count = 1
            return False

        sim = self._similarity(current_hash, self._baseline_hash)

        # If appended intensity bits indicate a strong divergence, force a reset
        if self.extra_intensity_bits:
            mask = (1 << self.extra_intensity_bits) - 1
            prev_intensity = self._baseline_hash & mask
            curr_intensity = current_hash & mask
            scale = max(1, (1 << self.extra_intensity_bits) - 1)
            intensity_delta = abs(prev_intensity - curr_intensity) / scale
            # Threshold chosen so black/white uniform frames definitely reset.
            if intensity_delta >= 0.5:
                self._baseline_hash = current_hash
                self._stable_count = 1
                return self._stable_count >= required_matches

        if sim >= self.similarity_threshold:
            self._stable_count += 1
        else:
            self._baseline_hash = current_hash
            self._stable_count = 1

        return self._stable_count >= required_matches

    def reset(self):
        """Reset hash tracking and stability state."""
        self._last_hash = None
        self._baseline_hash = None
        self._stable_count = 0
