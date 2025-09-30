"""Tests for dataset capture functionality."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

from azl_bot.core.dataset_capture import compute_dhash, hamming_distance, DatasetCapture


def test_compute_dhash():
    """Test dhash computation."""
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
    """Test Hamming distance calculation."""
    # Identical hashes
    assert hamming_distance("abcd1234", "abcd1234") == 0
    
    # Single bit difference (1 vs 0)
    assert hamming_distance("0000", "0001") == 1
    
    # Multiple differences
    assert hamming_distance("0000", "ffff") == 16  # All bits different
    
    # Different lengths (should return max)
    assert hamming_distance("00", "0000") >= 8


def test_dhash_similarity():
    """Test that similar images produce similar hashes."""
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


def test_dataset_capture_dedup():
    """Test deduplication in dataset capture."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "enabled": True,
            "sample_rate_hz": 100.0,  # High rate for testing
            "max_dim": 640,
            "format": "jpg",
            "jpeg_quality": 85,
            "dedupe": {
                "method": "dhash",
                "hamming_threshold": 3
            },
            "retention": {
                "max_files": 100,
                "max_days": 30
            },
            "metadata": True
        }
        
        capture = DatasetCapture(config, Path(tmpdir))
        
        # Create test image
        image1 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # First capture should succeed
        result1 = capture.capture(image1, (0, 0, 640, 480), (640, 480))
        assert result1 is not None
        assert result1.exists()
        
        # Immediate duplicate should be rejected
        result2 = capture.capture(image1, (0, 0, 640, 480), (640, 480))
        assert result2 is None, "Duplicate should be rejected"
        
        # Wait for sample rate interval
        import time
        time.sleep(0.02)  # Wait 20ms (rate is 100 Hz = 10ms interval)
        
        # Different image should be captured
        image2 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result3 = capture.capture(image2, (0, 0, 640, 480), (640, 480))
        assert result3 is not None
        assert result3.exists()
        
        # Check session count
        assert capture.get_session_count() == 2


def test_dataset_capture_metadata():
    """Test metadata saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "enabled": True,
            "sample_rate_hz": 100.0,
            "max_dim": 640,
            "format": "jpg",
            "jpeg_quality": 85,
            "dedupe": {
                "method": "dhash",
                "hamming_threshold": 3
            },
            "retention": {
                "max_files": 100,
                "max_days": 30
            },
            "metadata": True
        }
        
        capture = DatasetCapture(config, Path(tmpdir))
        
        # Create and capture image
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        context = {"screen": "home", "task": "test"}
        
        result = capture.capture(image, (0, 0, 640, 480), (640, 480), context=context)
        assert result is not None
        
        # Find metadata file
        metadata_files = list(result.parent.glob("*.json"))
        assert len(metadata_files) > 0, "Metadata file should exist"
        
        import json
        with open(metadata_files[0]) as f:
            metadata = json.load(f)
        
        # Check metadata structure
        assert "timestamp" in metadata
        assert "hash" in metadata
        assert "frame_size" in metadata
        assert metadata["frame_size"]["full_width"] == 640
        assert "active_rect" in metadata
        assert "captured_size" in metadata
        assert "context" in metadata
        assert metadata["context"]["screen"] == "home"


def test_dataset_capture_disabled():
    """Test that capture doesn't run when disabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "enabled": False,
            "sample_rate_hz": 100.0,
            "max_dim": 640,
            "format": "jpg",
            "jpeg_quality": 85,
            "dedupe": {
                "method": "dhash",
                "hamming_threshold": 3
            },
            "retention": {
                "max_files": 100,
                "max_days": 30
            },
            "metadata": True
        }
        
        capture = DatasetCapture(config, Path(tmpdir))
        
        # Create test image
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # Capture should return None when disabled
        result = capture.capture(image, (0, 0, 640, 480), (640, 480))
        assert result is None
        assert capture.get_session_count() == 0


def test_dataset_capture_toggle():
    """Test toggling capture on/off."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "enabled": False,
            "sample_rate_hz": 100.0,
            "max_dim": 640,
            "format": "jpg",
            "jpeg_quality": 85,
            "dedupe": {
                "method": "dhash",
                "hamming_threshold": 3
            },
            "retention": {
                "max_files": 100,
                "max_days": 30
            },
            "metadata": True
        }
        
        capture = DatasetCapture(config, Path(tmpdir))
        assert not capture.enabled
        
        # Toggle on
        capture.toggle_enabled()
        assert capture.enabled
        
        # Toggle off
        capture.toggle_enabled()
        assert not capture.enabled


if __name__ == "__main__":
    # Run tests
    test_compute_dhash()
    test_hamming_distance()
    test_dhash_similarity()
    test_dataset_capture_dedup()
    test_dataset_capture_metadata()
    test_dataset_capture_disabled()
    test_dataset_capture_toggle()
    print("All dataset capture tests passed!")
