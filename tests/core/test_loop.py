"""Unit tests for StateLoop implementation."""

import time
from typing import Optional
from unittest.mock import Mock, MagicMock

import numpy as np
import pytest

from azl_bot.core.loop import StateLoop, LoopConfig, LoopMetrics
from azl_bot.core.hashing import FrameHasher


class MockFrame:
    """Mock Frame for testing."""
    
    def __init__(self, image_data: Optional[np.ndarray] = None):
        """Initialize mock frame."""
        if image_data is None:
            # Create a default 100x100 random image
            image_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.image_bgr = image_data
        self.active_rect = (0, 0, 100, 100)
        self.full_w = 100
        self.full_h = 100


class MockCapture:
    """Mock Capture component."""
    
    def __init__(self, frame_sequence: Optional[list[MockFrame]] = None):
        """Initialize mock capture.
        
        Args:
            frame_sequence: List of frames to return in sequence. If None, returns random frames.
        """
        self.frame_sequence = frame_sequence or []
        self.call_count = 0
    
    def grab(self) -> MockFrame:
        """Return next frame in sequence or random frame."""
        if self.frame_sequence and self.call_count < len(self.frame_sequence):
            frame = self.frame_sequence[self.call_count]
            self.call_count += 1
            return frame
        # Return random frame if no sequence
        return MockFrame()


class MockActuator:
    """Mock Actuator component."""
    
    def __init__(self):
        """Initialize mock actuator."""
        self.tap_calls = []
        self.swipe_calls = []
    
    def tap_norm(self, x: float, y: float, active_rect: Optional[tuple[int, int, int, int]] = None) -> None:
        """Record tap."""
        self.tap_calls.append((x, y, active_rect))
    
    def swipe_norm(self, x1: float, y1: float, x2: float, y2: float, ms: int = 200, 
                   active_rect: Optional[tuple[int, int, int, int]] = None) -> None:
        """Record swipe."""
        self.swipe_calls.append((x1, y1, x2, y2, ms, active_rect))


class MockDevice:
    """Mock Device component."""
    
    def __init__(self):
        """Initialize mock device."""
        self.back_calls = 0
        self.home_calls = 0
    
    def key_back(self) -> None:
        """Record back button press."""
        self.back_calls += 1
    
    def key_home(self) -> None:
        """Record home button press."""
        self.home_calls += 1


class TestLoopConfig:
    """Test LoopConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LoopConfig()
        
        assert config.target_fps == 2.0
        assert config.stability_frames == 3
        assert config.stability_timeout_sec == 10.0
        assert config.max_retries == 3
        assert config.retry_backoff_base == 1.5
        assert config.recovery_enabled is True
        assert config.hamming_threshold == 0.05
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = LoopConfig(
            target_fps=5.0,
            stability_frames=5,
            max_retries=5,
            recovery_enabled=False
        )
        
        assert config.target_fps == 5.0
        assert config.stability_frames == 5
        assert config.max_retries == 5
        assert config.recovery_enabled is False


class TestLoopMetrics:
    """Test LoopMetrics dataclass."""
    
    def test_initial_metrics(self):
        """Test initial metric values."""
        metrics = LoopMetrics()
        
        assert metrics.actions_attempted == 0
        assert metrics.actions_succeeded == 0
        assert metrics.actions_retried == 0
        assert metrics.recoveries_triggered == 0
        assert metrics.failures == 0
        assert metrics.avg_resolve_time == 0.0
    
    def test_avg_resolve_time_calculation(self):
        """Test average resolve time calculation."""
        metrics = LoopMetrics()
        
        metrics.total_resolve_time_sec = 10.0
        metrics.resolve_count = 5
        
        assert metrics.avg_resolve_time == 2.0
    
    def test_avg_resolve_time_with_zero_count(self):
        """Test average resolve time with no samples."""
        metrics = LoopMetrics()
        
        assert metrics.avg_resolve_time == 0.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = LoopMetrics(
            actions_attempted=10,
            actions_succeeded=8,
            actions_retried=2,
            recoveries_triggered=1,
            failures=2
        )
        
        d = metrics.to_dict()
        
        assert isinstance(d, dict)
        assert d["actions_attempted"] == 10
        assert d["actions_succeeded"] == 8
        assert d["actions_retried"] == 2
        assert d["recoveries_triggered"] == 1
        assert d["failures"] == 2


class TestStateLoop:
    """Test StateLoop class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = LoopConfig(
            target_fps=10.0,  # Fast for testing
            stability_frames=2,
            stability_timeout_sec=2.0,
            max_retries=2
        )
        self.capture = MockCapture()
        self.actuator = MockActuator()
        self.device = MockDevice()
        self.hasher = FrameHasher(hash_size=8, similarity_threshold=0.95)
        
        self.loop = StateLoop(
            config=self.config,
            capture=self.capture,
            actuator=self.actuator,
            device=self.device,
            hasher=self.hasher
        )
    
    def test_initialization(self):
        """Test StateLoop initialization."""
        assert self.loop.config == self.config
        assert self.loop.capture == self.capture
        assert self.loop.actuator == self.actuator
        assert self.loop.device == self.device
        assert self.loop.hasher == self.hasher
        assert isinstance(self.loop.metrics, LoopMetrics)
    
    def test_wait_for_stability_with_stable_frames(self):
        """Test stability detection with stable frame sequence."""
        # Create identical frames
        base_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        frames = [MockFrame(base_image.copy()) for _ in range(5)]
        
        self.capture.frame_sequence = frames
        
        # Should achieve stability
        stable, frame = self.loop.wait_for_stability(timeout_sec=5.0, required_frames=2)
        
        assert stable is True
        assert frame is not None
    
    def test_wait_for_stability_timeout(self):
        """Test stability timeout with changing frames."""
        # Create different frames
        frames = [MockFrame(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)) 
                  for _ in range(20)]
        
        self.capture.frame_sequence = frames
        
        # Should timeout
        stable, frame = self.loop.wait_for_stability(timeout_sec=0.5, required_frames=2)
        
        assert stable is False
        assert frame is not None
    
    def test_verify_action_success(self):
        """Test action verification with postcondition."""
        # Create stable frames for post-action
        base_image = np.ones((100, 100, 3), dtype=np.uint8) * 200
        frames = [MockFrame(base_image.copy()) for _ in range(10)]
        self.capture.frame_sequence = frames
        
        # Postcondition that always succeeds
        postcondition = lambda frame: True
        
        success, frame = self.loop.verify_action(postcondition=postcondition)
        
        assert success is True
        assert frame is not None
    
    def test_verify_action_postcondition_failure(self):
        """Test action verification with failing postcondition."""
        # Create stable frames
        base_image = np.ones((100, 100, 3), dtype=np.uint8) * 200
        frames = [MockFrame(base_image.copy()) for _ in range(10)]
        self.capture.frame_sequence = frames
        
        # Postcondition that always fails
        postcondition = lambda frame: False
        
        success, frame = self.loop.verify_action(postcondition=postcondition)
        
        assert success is False
    
    def test_execute_with_retry_success_first_attempt(self):
        """Test action execution succeeding on first attempt."""
        # Create stable frames
        base_image = np.ones((100, 100, 3), dtype=np.uint8) * 150
        frames = [MockFrame(base_image.copy()) for _ in range(20)]
        self.capture.frame_sequence = frames
        
        action_called = []
        def test_action():
            action_called.append(True)
        
        success, frame = self.loop.execute_with_retry(test_action, max_retries=2)
        
        assert success is True
        assert len(action_called) == 1
        assert self.loop.metrics.actions_attempted >= 1
        assert self.loop.metrics.actions_succeeded == 1
    
    def test_execute_with_retry_backoff_timing(self):
        """Test exponential backoff timing between retries."""
        # Create stable frames
        base_image = np.ones((100, 100, 3), dtype=np.uint8) * 150
        frames = [MockFrame(base_image.copy()) for _ in range(100)]
        self.capture.frame_sequence = frames
        
        attempt_times = []
        def test_action():
            attempt_times.append(time.time())
        
        # Postcondition that fails first 2 times, succeeds on 3rd
        call_count = [0]
        def postcondition(frame):
            call_count[0] += 1
            return call_count[0] >= 3
        
        start_time = time.time()
        success, frame = self.loop.execute_with_retry(test_action, postcondition=postcondition, max_retries=3)
        
        assert success is True
        assert len(attempt_times) == 3
        
        # Check backoff timing (should be approximately 1.5^1 and 1.5^2)
        if len(attempt_times) >= 3:
            delay1 = attempt_times[1] - attempt_times[0]
            delay2 = attempt_times[2] - attempt_times[1]
            
            # Backoff should be approximately 1.5s and 2.25s (with some tolerance)
            assert delay1 >= 1.3  # 1.5^1 = 1.5 with tolerance
            assert delay2 >= 2.0  # 1.5^2 = 2.25 with tolerance
    
    def test_execute_with_retry_failure_after_max_attempts(self):
        """Test action execution failing after max retries."""
        # Create stable frames for pre-action
        base_image = np.ones((100, 100, 3), dtype=np.uint8) * 150
        frames = [MockFrame(base_image.copy()) for _ in range(50)]
        self.capture.frame_sequence = frames
        
        action_called = []
        def test_action():
            action_called.append(True)
        
        # Postcondition that always fails
        postcondition = lambda frame: False
        
        success, frame = self.loop.execute_with_retry(test_action, postcondition=postcondition, max_retries=2)
        
        assert success is False
        assert len(action_called) == 3  # initial + 2 retries
        assert self.loop.metrics.failures == 1
        assert self.loop.metrics.actions_retried >= 2
    
    def test_recovery_sequence(self):
        """Test recovery sequence execution."""
        result = self.loop.recovery()
        
        assert result is True
        # Default recovery: 3 backs + 1 home
        assert self.device.back_calls == 3
        assert self.device.home_calls == 1
        assert self.loop.metrics.recoveries_triggered == 1
    
    def test_recovery_custom_sequence(self):
        """Test custom recovery sequence."""
        custom_sequence = ["back", "back", "home"]
        result = self.loop.recovery(recovery_sequence=custom_sequence)
        
        assert result is True
        assert self.device.back_calls == 2
        assert self.device.home_calls == 1
    
    def test_recovery_disabled(self):
        """Test recovery when disabled in config."""
        self.config.recovery_enabled = False
        result = self.loop.recovery()
        
        assert result is False
        assert self.device.back_calls == 0
        assert self.device.home_calls == 0
    
    def test_run_action_with_recovery_triggers_recovery_on_failure(self):
        """Test that recovery is triggered when action fails."""
        # Create stable frames
        base_image = np.ones((100, 100, 3), dtype=np.uint8) * 150
        frames = [MockFrame(base_image.copy()) for _ in range(100)]
        self.capture.frame_sequence = frames
        
        def test_action():
            pass
        
        # Postcondition that always fails
        postcondition = lambda frame: False
        
        success, frame = self.loop.run_action_with_recovery(test_action, postcondition=postcondition)
        
        assert success is False
        # Recovery should have been triggered
        assert self.loop.metrics.recoveries_triggered >= 1
        assert self.device.back_calls >= 3
    
    def test_get_metrics(self):
        """Test metrics retrieval."""
        self.loop.metrics.actions_attempted = 5
        self.loop.metrics.actions_succeeded = 3
        
        metrics = self.loop.get_metrics()
        
        assert isinstance(metrics, dict)
        assert metrics["actions_attempted"] == 5
        assert metrics["actions_succeeded"] == 3
    
    def test_reset_metrics(self):
        """Test metrics reset."""
        self.loop.metrics.actions_attempted = 10
        self.loop.metrics.failures = 5
        
        self.loop.reset_metrics()
        
        assert self.loop.metrics.actions_attempted == 0
        assert self.loop.metrics.failures == 0


class TestStabilityWindows:
    """Test stability window detection."""
    
    def test_stability_with_synthetic_frames(self):
        """Test stability detection with controlled frame sequence."""
        hasher = FrameHasher(hash_size=8, similarity_threshold=0.95)
        
        # Create a stable sequence
        base_image = np.ones((100, 100, 3), dtype=np.uint8) * 100
        
        # First frame - not stable yet (only 1 frame)
        assert hasher.is_stable(base_image, required_matches=3) is False
        
        # Second frame - still not stable (only 2 frames)
        assert hasher.is_stable(base_image.copy(), required_matches=3) is False
        
        # Third frame - should be stable now (3 matching frames)
        assert hasher.is_stable(base_image.copy(), required_matches=3) is True
        
        # Fourth frame - should remain stable
        assert hasher.is_stable(base_image.copy(), required_matches=3) is True
    
    def test_stability_breaks_on_change(self):
        """Test that stability breaks when frame changes."""
        hasher = FrameHasher(hash_size=8, similarity_threshold=0.95)
        
        # Create stable sequence
        base_image = np.ones((100, 100, 3), dtype=np.uint8) * 100
        
        # Build up stability
        hasher.is_stable(base_image, required_matches=3)
        hasher.is_stable(base_image.copy(), required_matches=3)
        assert hasher.is_stable(base_image.copy(), required_matches=3) is True
        
        # Introduce a different frame
        different_image = np.ones((100, 100, 3), dtype=np.uint8) * 200
        assert hasher.is_stable(different_image, required_matches=3) is False
        
        # Should need to rebuild stability
        assert hasher.is_stable(different_image.copy(), required_matches=3) is False


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""
    
    def test_complete_action_cycle(self):
        """Test complete Sense→Think→Act→Check cycle."""
        config = LoopConfig(
            target_fps=10.0,
            stability_frames=2,
            stability_timeout_sec=2.0,
            max_retries=1
        )
        
        # Create frame sequence: pre-stable, post-action stable
        base_image = np.ones((100, 100, 3), dtype=np.uint8) * 100
        frames = [MockFrame(base_image.copy()) for _ in range(20)]
        
        capture = MockCapture(frames)
        actuator = MockActuator()
        device = MockDevice()
        
        loop = StateLoop(config, capture, actuator, device)
        
        # Execute action
        action_executed = [False]
        def test_action():
            action_executed[0] = True
            actuator.tap_norm(0.5, 0.5)
        
        success, frame = loop.execute_with_retry(test_action)
        
        assert success is True
        assert action_executed[0] is True
        assert len(actuator.tap_calls) == 1
        assert loop.metrics.actions_attempted >= 1
        assert loop.metrics.actions_succeeded == 1
    
    def test_action_retry_then_success(self):
        """Test action failing once then succeeding on retry."""
        config = LoopConfig(
            target_fps=10.0,
            stability_frames=2,
            stability_timeout_sec=2.0,
            max_retries=3
        )
        
        base_image = np.ones((100, 100, 3), dtype=np.uint8) * 100
        frames = [MockFrame(base_image.copy()) for _ in range(50)]
        
        capture = MockCapture(frames)
        actuator = MockActuator()
        device = MockDevice()
        
        loop = StateLoop(config, capture, actuator, device)
        
        # Postcondition fails first time, succeeds second time
        attempt_count = [0]
        def postcondition(frame):
            attempt_count[0] += 1
            return attempt_count[0] >= 2
        
        def test_action():
            pass
        
        success, frame = loop.execute_with_retry(test_action, postcondition=postcondition)
        
        assert success is True
        assert attempt_count[0] == 2
        assert loop.metrics.actions_retried >= 1
        assert loop.metrics.actions_succeeded == 1
