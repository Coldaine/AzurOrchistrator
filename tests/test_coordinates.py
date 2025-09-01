"""Unit tests for coordinate transformation pipeline."""

import pytest
import numpy as np
from unittest.mock import Mock

from azl_bot.core.capture import Frame, Capture
from azl_bot.core.actuator import Actuator
from azl_bot.core.device import Device


class TestCoordinateTransforms:
    """Test coordinate transformation pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock device
        self.mock_device = Mock(spec=Device)
        self.mock_device.info.return_value = {"width": 1920, "height": 1080, "density": 480}

        # Create capture instance
        self.capture = Capture(self.mock_device)

        # Create actuator instance
        self.actuator = Actuator(self.mock_device)

    def test_norm_to_pixels_basic(self):
        """Test basic normalized to pixel conversion."""
        # Create test frame with active area
        frame = Frame(
            png_bytes=b"fake_png",
            image_bgr=np.zeros((800, 1200, 3), dtype=np.uint8),
            full_w=1920,
            full_h=1080,
            active_rect=(60, 132, 1800, 816),  # Letterboxed 1080p
            ts=0.0
        )

        # Test center point
        x_norm, y_norm = 0.5, 0.5
        x_pixel, y_pixel = self.capture.norm_to_pixels(x_norm, y_norm, frame)

        # Center of active area should be at (60+900, 132+408) = (960, 540)
        # which is center of 1920x1080 screen
        assert x_pixel == 960
        assert y_pixel == 540

    def test_norm_to_pixels_corners(self):
        """Test corner coordinate conversions."""
        frame = Frame(
            png_bytes=b"fake_png",
            image_bgr=np.zeros((800, 1200, 3), dtype=np.uint8),
            full_w=1920,
            full_h=1080,
            active_rect=(60, 132, 1800, 816),
            ts=0.0
        )

        # Top-left corner (0,0)
        x_pixel, y_pixel = self.capture.norm_to_pixels(0.0, 0.0, frame)
        assert x_pixel == 60   # active_rect x
        assert y_pixel == 132  # active_rect y

        # Bottom-right corner (1,1)
        x_pixel, y_pixel = self.capture.norm_to_pixels(1.0, 1.0, frame)
        assert x_pixel == 60 + 1800  # active_rect x + width
        assert y_pixel == 132 + 816  # active_rect y + height

    def test_pixels_to_norm_basic(self):
        """Test basic pixel to normalized conversion."""
        frame = Frame(
            png_bytes=b"fake_png",
            image_bgr=np.zeros((800, 1200, 3), dtype=np.uint8),
            full_w=1920,
            full_h=1080,
            active_rect=(60, 132, 1800, 816),
            ts=0.0
        )

        # Center pixel
        x_pixel, y_pixel = 960, 540
        x_norm, y_norm = self.capture.pixels_to_norm(x_pixel, y_pixel, frame)

        assert abs(x_norm - 0.5) < 0.001
        assert abs(y_norm - 0.5) < 0.001

    def test_pixels_to_norm_corners(self):
        """Test corner pixel conversions."""
        frame = Frame(
            png_bytes=b"fake_png",
            image_bgr=np.zeros((800, 1200, 3), dtype=np.uint8),
            full_w=1920,
            full_h=1080,
            active_rect=(60, 132, 1800, 816),
            ts=0.0
        )

        # Top-left corner
        x_norm, y_norm = self.capture.pixels_to_norm(60, 132, frame)
        assert abs(x_norm - 0.0) < 0.001
        assert abs(y_norm - 0.0) < 0.001

        # Bottom-right corner
        x_norm, y_norm = self.capture.pixels_to_norm(60 + 1800, 132 + 816, frame)
        assert abs(x_norm - 1.0) < 0.001
        assert abs(y_norm - 1.0) < 0.001

    def test_round_trip_conversion(self):
        """Test that norm->pixel->norm produces original values."""
        frame = Frame(
            png_bytes=b"fake_png",
            image_bgr=np.zeros((800, 1200, 3), dtype=np.uint8),
            full_w=1920,
            full_h=1080,
            active_rect=(60, 132, 1800, 816),
            ts=0.0
        )

        test_points = [
            (0.0, 0.0),
            (0.5, 0.5),
            (1.0, 1.0),
            (0.25, 0.75),
            (0.9, 0.1)
        ]

        for x_norm, y_norm in test_points:
            # Convert norm -> pixel
            x_pixel, y_pixel = self.capture.norm_to_pixels(x_norm, y_norm, frame)

            # Convert pixel -> norm
            x_norm_round, y_norm_round = self.capture.pixels_to_norm(x_pixel, y_pixel, frame)

            # Should be very close to original (within 1 pixel precision)
            assert abs(x_norm - x_norm_round) < 0.001
            assert abs(y_norm - y_norm_round) < 0.001

    def test_actuator_tap_norm_with_active_rect(self):
        """Test actuator tap_norm with active_rect parameter."""
        # Test with active rect
        active_rect = (60, 132, 1800, 816)

        # Mock the device._adb call
        self.mock_device._adb = Mock()

        # Test tap at center
        self.actuator.tap_norm(0.5, 0.5, active_rect)

        # Should call ADB with correct pixel coordinates
        self.mock_device._adb.assert_called_with("shell", "input", "tap", "960", "540")

    def test_actuator_tap_norm_without_active_rect(self):
        """Test actuator tap_norm fallback without active_rect."""
        # Mock the device._adb call
        self.mock_device._adb = Mock()

        # Test tap without active_rect (should use full device)
        self.actuator.tap_norm(0.5, 0.5)

        # Should call ADB with device center coordinates
        self.mock_device._adb.assert_called_with("shell", "input", "tap", "960", "540")

    def test_actuator_swipe_norm_with_active_rect(self):
        """Test actuator swipe_norm with active_rect parameter."""
        active_rect = (60, 132, 1800, 816)

        # Mock the device._adb call
        self.mock_device._adb = Mock()

        # Test swipe from center to right
        self.actuator.swipe_norm(0.5, 0.5, 0.8, 0.5, active_rect=active_rect)

        # Should call ADB with correct pixel coordinates
        expected_calls = self.mock_device._adb.call_args_list
        assert len(expected_calls) == 1

        args = expected_calls[0][0]
        assert args[0] == "shell"
        assert args[1] == "input"
        assert args[2] == "swipe"
        assert args[3] == "960"  # Start X
        assert args[4] == "540"  # Start Y
        assert args[5] == "1500"  # End X (60 + 0.8 * 1800)
        assert args[6] == "540"  # End Y

    def test_coordinate_bounds_clamping(self):
        """Test that coordinates are properly clamped to screen bounds."""
        active_rect = (60, 132, 1800, 816)

        # Mock the device._adb call
        self.mock_device._adb = Mock()

        # Test coordinates outside bounds
        self.actuator.tap_norm(-0.1, 1.5, active_rect)  # Outside bounds

        # Should still call ADB with clamped coordinates
        call_args = self.mock_device._adb.call_args
        assert call_args is not None

        # Coordinates should be clamped to valid range
        x_coord = int(call_args[0][3])
        y_coord = int(call_args[0][4])

        assert 0 <= x_coord <= 1919  # Within screen bounds
        assert 0 <= y_coord <= 1079

    def test_different_resolutions(self):
        """Test coordinate transforms with different device resolutions."""
        test_resolutions = [
            (1920, 1080),  # 1080p
            (1280, 720),   # 720p
            (2560, 1440),  # 1440p
            (3840, 2160),  # 4K
        ]

        for width, height in test_resolutions:
            # Create mock device with this resolution
            mock_device = Mock(spec=Device)
            mock_device.info.return_value = {"width": width, "height": height, "density": 480}

            actuator = Actuator(mock_device)

            # Test center tap
            mock_device._adb = Mock()
            actuator.tap_norm(0.5, 0.5)

            # Should tap at center of screen
            call_args = mock_device._adb.call_args
            x_coord = int(call_args[0][3])
            y_coord = int(call_args[0][4])

            assert x_coord == width // 2
            assert y_coord == height // 2

    def test_letterbox_detection(self):
        """Test letterbox detection with various aspect ratios."""
        # Create test images with different letterboxing scenarios

        # 16:9 content in 4:3 frame (horizontal letterboxing)
        img_4_3 = np.zeros((1080, 1440, 3), dtype=np.uint8)  # 4:3 = 1.333
        # Add black bars on sides
        active_rect = self.capture.detect_letterbox(img_4_3)
        assert active_rect[2] > 0  # Width should be detected
        assert active_rect[3] == 1080  # Full height

        # 4:3 content in 16:9 frame (vertical letterboxing)
        img_16_9 = np.zeros((1080, 1920, 3), dtype=np.uint8)  # 16:9 = 1.777
        # Add black bars on top/bottom
        active_rect = self.capture.detect_letterbox(img_16_9)
        assert active_rect[3] > 0  # Height should be detected
        assert active_rect[2] == 1920  # Full width

    def test_empty_active_area(self):
        """Test handling of empty or invalid active areas."""
        # Test with zero-sized active area
        frame = Frame(
            png_bytes=b"fake_png",
            image_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
            full_w=1920,
            full_h=1080,
            active_rect=(0, 0, 0, 0),  # Empty active area
            ts=0.0
        )

        # Should handle gracefully
        x_pixel, y_pixel = self.capture.norm_to_pixels(0.5, 0.5, frame)
        # Should still produce valid coordinates
        assert isinstance(x_pixel, int)
        assert isinstance(y_pixel, int)

    def test_coordinate_precision(self):
        """Test coordinate conversion precision."""
        frame = Frame(
            png_bytes=b"fake_png",
            image_bgr=np.zeros((800, 1200, 3), dtype=np.uint8),
            full_w=1920,
            full_h=1080,
            active_rect=(60, 132, 1800, 816),
            ts=0.0
        )

        # Test various normalized coordinates
        test_cases = [
            (0.0, 0.0, 60, 132),
            (0.25, 0.25, 60 + 450, 132 + 204),
            (0.5, 0.5, 60 + 900, 132 + 408),
            (0.75, 0.75, 60 + 1350, 132 + 612),
            (1.0, 1.0, 60 + 1800, 132 + 816),
        ]

        for x_norm, y_norm, expected_x, expected_y in test_cases:
            x_pixel, y_pixel = self.capture.norm_to_pixels(x_norm, y_norm, frame)
            assert x_pixel == expected_x
            assert y_pixel == expected_y

    def test_actuator_coordinate_validation(self):
        """Test actuator coordinate validation and error handling."""
        # Test with invalid coordinates
        active_rect = (60, 132, 1800, 816)

        # Mock the device._adb call
        self.mock_device._adb = Mock()

        # Test with None active_rect (should not crash)
        self.actuator.tap_norm(0.5, 0.5, None)
        assert self.mock_device._adb.called

        # Test with invalid active_rect values
        invalid_rect = (0, 0, -100, -100)  # Negative dimensions
        self.actuator.tap_norm(0.5, 0.5, invalid_rect)
        # Should still work (coordinates will be clamped)

    def test_transform_pipeline_integration(self):
        """Test complete transform pipeline from norm to device."""
        # Create a complete pipeline test
        frame = Frame(
            png_bytes=b"fake_png",
            image_bgr=np.zeros((800, 1200, 3), dtype=np.uint8),
            full_w=1920,
            full_h=1080,
            active_rect=(60, 132, 1800, 816),
            ts=0.0
        )

        # Mock device ADB call
        self.mock_device._adb = Mock()

        # Start with normalized coordinates
        x_norm, y_norm = 0.5, 0.5

        # Convert through pipeline: norm -> pixel -> device tap
        self.actuator.tap_norm(x_norm, y_norm, frame.active_rect)

        # Verify final device coordinates
        call_args = self.mock_device._adb.call_args
        x_device = int(call_args[0][3])
        y_device = int(call_args[0][4])

        # Should be center of screen
        assert x_device == 960
        assert y_device == 540

        # Verify round-trip consistency
        x_norm_round, y_norm_round = self.capture.pixels_to_norm(x_device, y_device, frame)
        assert abs(x_norm - x_norm_round) < 0.01
        assert abs(y_norm - y_norm_round) < 0.01