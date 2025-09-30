"""Tests for resolver harness."""

import numpy as np
import cv2
from pathlib import Path
import tempfile
import json

from azl_bot.tools.resolver_harness import (
    load_image_as_frame,
    draw_candidates_overlay,
    resolve_offline,
    save_results_csv,
    save_results_json
)
from azl_bot.core.resolver import Resolver, Candidate
from azl_bot.core.configs import create_default_config
from azl_bot.core.llm_client import Target


def create_test_image(width=640, height=480):
    """Create a simple test image."""
    image = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    # Add some features
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)
    cv2.circle(image, (400, 300), 50, (0, 255, 0), -1)
    cv2.putText(image, "TEST", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return image


def test_load_image_as_frame():
    """Test loading an image as a frame."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test image
        image = create_test_image()
        image_path = Path(tmpdir) / "test.png"
        cv2.imwrite(str(image_path), image)
        
        # Load as frame
        frame = load_image_as_frame(image_path)
        
        assert frame is not None
        assert frame.full_w == 640
        assert frame.full_h == 480
        assert frame.active_rect == (0, 0, 640, 480)
        assert frame.image_bgr.shape == (480, 640, 3)


def test_draw_candidates_overlay():
    """Test drawing overlay with candidates."""
    image = create_test_image()
    
    candidates = [
        Candidate((0.5, 0.5), 0.85, "ocr"),
        Candidate((0.3, 0.7), 0.72, "template"),
        Candidate((0.8, 0.2), 0.65, "orb")
    ]
    
    top_candidate = candidates[0]
    
    overlay = draw_candidates_overlay(image, candidates, top_candidate, 640, 480)
    
    assert overlay.shape == image.shape
    # Check that overlay is different from original (has drawings)
    assert not np.array_equal(overlay, image)


def test_save_results_csv():
    """Test CSV output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from azl_bot.tools.resolver_harness import ResolveResult
        
        results = [
            ResolveResult(
                image_path="test1.png",
                target_kind="text",
                target_value="TEST",
                candidates=[
                    {"method": "ocr", "x_norm": 0.5, "y_norm": 0.5, "confidence": 0.85}
                ],
                top_candidate={"method": "ocr", "x_norm": 0.5, "y_norm": 0.5, "confidence": 0.85},
                resolve_time_ms=123.4,
                method_times={"text": 120.0}
            )
        ]
        
        csv_path = Path(tmpdir) / "results.csv"
        save_results_csv(results, csv_path)
        
        assert csv_path.exists()
        
        # Check content
        with open(csv_path) as f:
            lines = f.readlines()
        
        assert len(lines) == 2  # Header + 1 data row
        assert "image_path" in lines[0]
        assert "test1.png" in lines[1]


def test_save_results_json():
    """Test JSON output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from azl_bot.tools.resolver_harness import ResolveResult
        
        results = [
            ResolveResult(
                image_path="test1.png",
                target_kind="text",
                target_value="TEST",
                candidates=[
                    {"method": "ocr", "x_norm": 0.5, "y_norm": 0.5, "confidence": 0.85}
                ],
                top_candidate={"method": "ocr", "x_norm": 0.5, "y_norm": 0.5, "confidence": 0.85},
                resolve_time_ms=123.4,
                method_times={"text": 120.0}
            )
        ]
        
        json_path = Path(tmpdir) / "results.json"
        save_results_json(results, json_path)
        
        assert json_path.exists()
        
        # Check content
        with open(json_path) as f:
            data = json.load(f)
        
        assert len(data) == 1
        assert data[0]["image_path"] == "test1.png"
        assert data[0]["target_kind"] == "text"


def test_harness_integration():
    """Test end-to-end harness workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test image with text
        image = create_test_image()
        image_path = Path(tmpdir) / "test.png"
        cv2.imwrite(str(image_path), image)
        
        # Create a mock OCR client
        class MockOCR:
            def extract_text(self, frame):
                # Return mock OCR result
                return [
                    {
                        "text": "TEST",
                        "bbox": [280, 80, 380, 120],
                        "confidence": 0.9
                    }
                ]
        
        # Create resolver
        config = create_default_config()
        resolver = Resolver(
            config=config.resolver.model_dump(),
            ocr_client=MockOCR(),
            templates_dir=str(Path(tmpdir) / "templates"),
            llm=None
        )
        
        # Load frame
        frame = load_image_as_frame(image_path)
        
        # Create target
        target = Target(kind="text", value="TEST")
        
        # Resolve
        result = resolve_offline(resolver, frame, target)
        
        assert result is not None
        assert result.target_kind == "text"
        assert result.target_value == "TEST"
        assert len(result.candidates) > 0
        assert result.top_candidate is not None
        assert result.resolve_time_ms > 0


if __name__ == "__main__":
    # Run tests
    test_load_image_as_frame()
    test_draw_candidates_overlay()
    test_save_results_csv()
    test_save_results_json()
    test_harness_integration()
    print("All resolver harness tests passed!")
