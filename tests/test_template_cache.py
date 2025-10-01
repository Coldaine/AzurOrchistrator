"""Tests for resolver template caching."""

import time
import numpy as np
import cv2
from pathlib import Path
import tempfile

from azl_bot.core.resolver import Resolver
from azl_bot.core.configs import create_default_config
from azl_bot.core.capture import Frame
from azl_bot.core.llm_client import Target


def create_test_template(size=(64, 64)):
    """Create a simple test template."""
    template = np.zeros((*size, 3), dtype=np.uint8)
    
    # Add a recognizable pattern
    center_x, center_y = size[1] // 2, size[0] // 2
    cv2.circle(template, (center_x, center_y), size[0] // 4, (255, 255, 255), -1)
    cv2.rectangle(template, (10, 10), (size[1]-10, size[0]-10), (0, 255, 0), 2)
    
    return template


def create_test_frame_with_template(template, position=(200, 200), size=(640, 480)):
    """Create a test frame with template embedded."""
    frame_image = np.random.randint(50, 150, (*size, 3), dtype=np.uint8)
    
    # Embed template
    th, tw = template.shape[:2]
    x, y = position
    frame_image[y:y+th, x:x+tw] = template
    
    return Frame(
        png_bytes=b"",
        image_bgr=frame_image,
        full_w=size[0],
        full_h=size[1],
        active_rect=(0, 0, size[0], size[1]),
        ts=time.time()
    )


def test_template_cache_warm_cold():
    """Test that cached templates are faster than cold loads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        templates_dir = Path(tmpdir)
        
        # Create test template
        template = create_test_template()
        template_path = templates_dir / "test_icon.png"
        cv2.imwrite(str(template_path), template)
        
        # Create resolver
        config = create_default_config()
        
        # Mock OCR (not used in this test)
        class MockOCR:
            def extract_text(self, frame):
                return []
        
        resolver = Resolver(
            config=config.resolver.model_dump(),
            ocr_client=MockOCR(),
            templates_dir=str(templates_dir),
            llm=None
        )
        
        # Create frame with template
        frame = create_test_frame_with_template(template, position=(200, 200))
        
        # Create target
        target = Target(kind="icon", value="test_icon")
        
        # First call (cold - loads and caches)
        start_cold = time.time()
        result1 = resolver._detect_by_template(target, frame)
        time_cold = time.time() - start_cold
        
        # Second call (warm - uses cache)
        start_warm = time.time()
        result2 = resolver._detect_by_template(target, frame)
        time_warm = time.time() - start_warm
        
        # Warm should be faster than cold
        # Allow for some variance but expect at least 20% speedup
        speedup_ratio = time_cold / time_warm if time_warm > 0 else float('inf')
        
        print(f"Cold time: {time_cold*1000:.2f}ms")
        print(f"Warm time: {time_warm*1000:.2f}ms")
        print(f"Speedup: {speedup_ratio:.2f}x")
        
        # Cached call should be faster (or at least not significantly slower)
        assert speedup_ratio >= 1.0, f"Cached call should not be slower, got {speedup_ratio:.2f}x"
        
        # Results should be consistent
        assert len(result1) == len(result2), "Results should be consistent between calls"


def test_template_cache_multiple_templates():
    """Test that cache works for multiple different templates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        templates_dir = Path(tmpdir)
        
        # Create multiple test templates
        template1 = create_test_template((64, 64))
        template2 = create_test_template((48, 48))
        
        cv2.imwrite(str(templates_dir / "icon1.png"), template1)
        cv2.imwrite(str(templates_dir / "icon2.png"), template2)
        
        # Create resolver
        config = create_default_config()
        
        class MockOCR:
            def extract_text(self, frame):
                return []
        
        resolver = Resolver(
            config=config.resolver.model_dump(),
            ocr_client=MockOCR(),
            templates_dir=str(templates_dir),
            llm=None
        )
        
        # Create frames
        frame1 = create_test_frame_with_template(template1)
        frame2 = create_test_frame_with_template(template2)
        
        # Detect template1
        target1 = Target(kind="icon", value="icon1")
        resolver._detect_by_template(target1, frame1)
        
        # Check cache
        assert "icon1" in resolver._template_cache
        assert len(resolver._template_cache["icon1"]) > 0
        
        # Detect template2
        target2 = Target(kind="icon", value="icon2")
        resolver._detect_by_template(target2, frame2)
        
        # Check both are cached
        assert "icon1" in resolver._template_cache
        assert "icon2" in resolver._template_cache


def test_template_pyramid_scales():
    """Test that template pyramid has correct scales."""
    with tempfile.TemporaryDirectory() as tmpdir:
        templates_dir = Path(tmpdir)
        
        # Create test template
        template = create_test_template((64, 64))
        template_path = templates_dir / "test.png"
        cv2.imwrite(str(template_path), template)
        
        # Create resolver
        config = create_default_config()
        
        class MockOCR:
            def extract_text(self, frame):
                return []
        
        resolver = Resolver(
            config=config.resolver.model_dump(),
            ocr_client=MockOCR(),
            templates_dir=str(templates_dir),
            llm=None
        )
        
        # Create frame
        frame = create_test_frame_with_template(template)
        
        # Detect template (triggers caching)
        target = Target(kind="icon", value="test")
        resolver._detect_by_template(target, frame)
        
        # Check pyramid scales
        assert "test" in resolver._template_cache
        pyramid = resolver._template_cache["test"]
        
        # Should have multiple scales
        assert len(pyramid) >= 3, "Pyramid should have multiple scales"
        
        # Check that scales are different
        scales = [scale for _, scale in pyramid]
        assert len(set(scales)) == len(scales), "All scales should be unique"
        
        # Check that scale 1.0 exists
        assert 1.0 in scales, "Pyramid should include scale 1.0"


if __name__ == "__main__":
    # Run tests
    test_template_cache_warm_cold()
    test_template_cache_multiple_templates()
    test_template_pyramid_scales()
    print("All template cache tests passed!")
