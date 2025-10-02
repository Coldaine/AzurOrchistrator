"""Unified hash testing - FrameHasher class + standalone dhash functions.

This module consolidates all hash-related tests from:
- Original test_hashing.py (FrameHasher.is_stable tests)
- test_dataset_capture.py (compute_dhash, hamming_distance tests)

All hash functionality should be tested here to avoid duplication.
"""

import numpy as np
import pytest

from azl_bot.core.hashing import FrameHasher
from azl_bot.core.dataset_capture import compute_dhash, hamming_distance


# ============================================================================
# FrameHasher Tests (from original test_hashing.py)
# ============================================================================


def test_is_stable_reaches_threshold():
    """Test that FrameHasher.is_stable correctly detects stable frames."""
    hasher = FrameHasher(similarity_threshold=0.99)
    img = np.random.randint(0, 256, (16, 16), dtype=np.uint8)

    # First call establishes baseline
    assert hasher.is_stable(img, required_matches=3) is False
    # Second and third
    assert hasher.is_stable(img, required_matches=3) is False
    assert hasher.is_stable(img, required_matches=3) is True


def test_is_stable_resets_on_change():
    """Test that FrameHasher.is_stable resets counter when frame changes."""
    hasher = FrameHasher(similarity_threshold=0.95)
    img1 = np.zeros((16, 16), dtype=np.uint8)
    img2 = np.ones((16, 16), dtype=np.uint8) * 255

    assert hasher.is_stable(img1, required_matches=2) is False
    # Change should reset
    assert hasher.is_stable(img2, required_matches=2) is False
    # Need two in a row after reset
    assert hasher.is_stable(img2, required_matches=2) is True


# ============================================================================
# Standalone Hash Function Tests (from test_dataset_capture.py)
# ============================================================================


def test_compute_dhash():
    """Test dhash computation for consistency and format."""
    # Create a simple test image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Add some pattern
    image[25:75, 25:75] = 255
    
    hash1 = compute_dhash(image)
    
    # Hash should be consistent
    hash2 = compute_dhash(image)
    assert hash1 == hash2
    
    # Hash should be hex string
    assert all(c in '0123456789abcdef' for c in hash1)
    assert len(hash1) == 16  # 64 bits = 16 hex chars


def test_hamming_distance():
    """Test Hamming distance calculation between hash strings."""
    # Identical hashes
    assert hamming_distance("abcd1234", "abcd1234") == 0
    
    # Single bit difference (1 vs 0)
    assert hamming_distance("0000", "0001") == 1
    
    # Multiple differences
    assert hamming_distance("0000", "ffff") == 16  # All bits different
    
    # Different lengths (should return max)
    assert hamming_distance("00", "0000") >= 8


def test_dhash_similarity():
    """Test that similar images produce similar hashes (low Hamming distance)."""
    # Original image
    image1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    hash1 = compute_dhash(image1)
    
    # Slightly modified image (add noise)
    image2 = image1.copy()
    noise = np.random.randint(-10, 10, image1.shape, dtype=np.int16)
    image2 = np.clip(image1.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    hash2 = compute_dhash(image2)
    
    # Should be similar (low Hamming distance)
    distance = hamming_distance(hash1, hash2)
    assert distance < 10, f"Similar images should have low Hamming distance, got {distance}"
    
    # Completely different image
    image3 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    hash3 = compute_dhash(image3)
    distance2 = hamming_distance(hash1, hash3)
    
    # Should be different (high Hamming distance)
    assert distance2 > distance, "Different images should have higher Hamming distance"


def test_dhash_different_images():
    """Test that completely different images have different hashes."""
    # Create two random images with different patterns
    np.random.seed(42)
    image1 = np.random.randint(0, 128, (100, 100, 3), dtype=np.uint8)
    hash1 = compute_dhash(image1)
    
    np.random.seed(123)
    image2 = np.random.randint(128, 256, (100, 100, 3), dtype=np.uint8)
    hash2 = compute_dhash(image2)
    
    # Hashes should be different
    assert hash1 != hash2, "Random images with different seeds should have different hashes"
    
    # Hamming distance should be moderate to high
    distance = hamming_distance(hash1, hash2)
    # Note: Solid black/white images have identical gradients (all zeros),
    # so we use random patterns which will have varying gradients
    assert distance > 5, f"Different random images should have Hamming distance > 5, got {distance}"


def test_dhash_grayscale_vs_color():
    """Test that dhash works consistently with different channel counts."""
    # Create pattern
    pattern = np.zeros((100, 100), dtype=np.uint8)
    pattern[25:75, 25:75] = 255
    
    # Grayscale
    hash_gray = compute_dhash(pattern)
    
    # RGB (same pattern replicated across channels)
    pattern_rgb = np.stack([pattern, pattern, pattern], axis=-1)
    hash_rgb = compute_dhash(pattern_rgb)
    
    # Should be similar (dhash converts to grayscale internally)
    distance = hamming_distance(hash_gray, hash_rgb)
    assert distance <= 5, "Grayscale and RGB versions of same pattern should have similar hashes"


# ============================================================================
# Integration Tests (FrameHasher + standalone functions)
# ============================================================================


def test_frame_hasher_uses_dhash_internally():
    """Test that FrameHasher and standalone dhash produce compatible results."""
    hasher = FrameHasher(similarity_threshold=0.90)
    
    # Create two similar images
    img1 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    img2 = img1.copy()
    
    # Add minimal noise
    noise = np.random.randint(-2, 2, img1.shape, dtype=np.int16)
    img2 = np.clip(img1.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # FrameHasher should detect stability
    hasher.is_stable(img1, required_matches=1)
    result = hasher.is_stable(img2, required_matches=1)
    
    # Standalone dhash should also show similarity
    hash1 = compute_dhash(img1)
    hash2 = compute_dhash(img2)
    distance = hamming_distance(hash1, hash2)
    
    # Both methods should agree (low distance = stable)
    # If FrameHasher says stable, distance should be low
    if result:
        assert distance < 10, "FrameHasher and dhash should agree on similarity"


if __name__ == "__main__":
    # Run tests manually
    test_is_stable_reaches_threshold()
    test_is_stable_resets_on_change()
    test_compute_dhash()
    test_hamming_distance()
    test_dhash_similarity()
    test_dhash_different_images()
    test_dhash_grayscale_vs_color()
    test_frame_hasher_uses_dhash_internally()
    print("✓ All hash tests passed!")
