"""Tests for UI enhancements."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from azl_bot.ui.state import UIState
from azl_bot.ui.overlays import OverlayRenderer


def test_ui_state_overlay_toggles():
    """Test UIState overlay toggle tracking."""
    state = UIState()
    
    # Check default values
    assert state.show_ocr_boxes == False
    assert state.show_template_matches == False
    assert state.show_orb_keypoints == False
    assert state.show_regions == True
    assert state.show_candidates == True
    assert state.selected_candidate_index is None
    
    # Test toggling
    state.show_ocr_boxes = True
    assert state.show_ocr_boxes == True
    
    # Test candidate selection
    state.selected_candidate_index = 2
    assert state.selected_candidate_index == 2
    
    print("✓ UIState overlay toggles test passed")


def test_ui_state_candidates():
    """Test UIState candidate tracking."""
    state = UIState()
    
    # Add candidates
    candidates = [
        {"point": (0.5, 0.5), "confidence": 0.9, "method": "ocr"},
        {"point": (0.3, 0.3), "confidence": 0.7, "method": "template"},
    ]
    state.update_candidates(candidates)
    
    assert len(state.last_candidates) == 2
    assert state.last_candidates[0]["confidence"] == 0.9
    
    print("✓ UIState candidates test passed")


def test_overlay_renderer_basic():
    """Test basic overlay rendering."""
    renderer = OverlayRenderer()
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    
    # Test with no overlays
    overlay_data = {
        "show_regions": False,
        "show_candidates": False,
    }
    result = renderer.render_overlays(img, overlay_data)
    assert result.shape == img.shape
    
    print("✓ OverlayRenderer basic test passed")


def test_overlay_renderer_candidates():
    """Test candidate overlay rendering."""
    renderer = OverlayRenderer()
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    
    candidates = [
        {"point": (0.5, 0.5), "confidence": 0.9, "method": "ocr"},
        {"point": (0.3, 0.3), "confidence": 0.7, "method": "template"},
    ]
    
    overlay_data = {
        "show_candidates": True,
        "candidates": candidates,
    }
    result = renderer.render_overlays(img, overlay_data)
    assert result.shape == img.shape
    
    # Test with selection
    overlay_data["selected_candidate_index"] = 0
    result = renderer.render_overlays(img, overlay_data)
    assert result.shape == img.shape
    
    print("✓ OverlayRenderer candidates test passed")


def test_overlay_renderer_ocr():
    """Test OCR overlay rendering."""
    renderer = OverlayRenderer()
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    
    ocr_results = [
        {"box_norm": (0.1, 0.1, 0.2, 0.05), "text": "Test", "conf": 0.9},
    ]
    
    overlay_data = {
        "show_ocr_boxes": True,
        "ocr_results": ocr_results,
    }
    result = renderer.render_overlays(img, overlay_data)
    assert result.shape == img.shape
    
    print("✓ OverlayRenderer OCR test passed")


def test_overlay_renderer_templates():
    """Test template matches overlay rendering."""
    renderer = OverlayRenderer()
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    
    template_matches = [
        {"point": (0.5, 0.5), "confidence": 0.92, "template": "test_icon"},
    ]
    
    overlay_data = {
        "show_template_matches": True,
        "template_matches": template_matches,
    }
    result = renderer.render_overlays(img, overlay_data)
    assert result.shape == img.shape
    
    print("✓ OverlayRenderer templates test passed")


def test_overlay_renderer_orb():
    """Test ORB keypoints overlay rendering."""
    renderer = OverlayRenderer()
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    
    orb_keypoints = [
        {"point": (0.2, 0.2)},
        {"point": (0.8, 0.8)},
    ]
    
    overlay_data = {
        "show_orb_keypoints": True,
        "orb_keypoints": orb_keypoints,
    }
    result = renderer.render_overlays(img, overlay_data)
    assert result.shape == img.shape
    
    print("✓ OverlayRenderer ORB test passed")


def test_overlay_renderer_all():
    """Test all overlays enabled together."""
    renderer = OverlayRenderer()
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    
    overlay_data = {
        "show_regions": True,
        "show_candidates": True,
        "show_ocr_boxes": True,
        "show_template_matches": True,
        "show_orb_keypoints": True,
        "regions": {
            "top_bar": (0.0, 0.0, 1.0, 0.12),
            "center": (0.2, 0.12, 0.6, 0.73),
        },
        "candidates": [
            {"point": (0.5, 0.5), "confidence": 0.9, "method": "ocr"},
        ],
        "selected_candidate_index": 0,
        "ocr_results": [
            {"box_norm": (0.1, 0.1, 0.2, 0.05), "text": "Test", "conf": 0.9},
        ],
        "template_matches": [
            {"point": (0.5, 0.5), "confidence": 0.92, "template": "test"},
        ],
        "orb_keypoints": [
            {"point": (0.2, 0.2)},
        ],
    }
    
    result = renderer.render_overlays(img, overlay_data)
    assert result.shape == img.shape
    
    print("✓ OverlayRenderer all overlays test passed")


def test_new_methods_exist():
    """Test that new methods exist on OverlayRenderer."""
    renderer = OverlayRenderer()
    
    assert hasattr(renderer, '_draw_template_matches')
    assert hasattr(renderer, '_draw_orb_keypoints')
    assert callable(renderer._draw_template_matches)
    assert callable(renderer._draw_orb_keypoints)
    
    print("✓ New methods exist test passed")


if __name__ == "__main__":
    print("Running UI enhancement tests...\n")
    
    test_ui_state_overlay_toggles()
    test_ui_state_candidates()
    test_overlay_renderer_basic()
    test_overlay_renderer_candidates()
    test_overlay_renderer_ocr()
    test_overlay_renderer_templates()
    test_overlay_renderer_orb()
    test_overlay_renderer_all()
    test_new_methods_exist()
    
    print("\n✅ All tests passed!")
