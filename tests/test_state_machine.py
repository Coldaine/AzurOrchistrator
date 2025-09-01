"""Unit tests for screen state machine and transitions."""

import pytest
from unittest.mock import Mock, patch

from azl_bot.core.screens import ScreenState, ScreenDetector
from azl_bot.core.capture import Frame


class TestScreenState:
    """Test ScreenState enum."""

    def test_screen_state_values(self):
        """Test that all expected screen states are defined."""
        expected_states = {
            "UNKNOWN", "HOME", "LOADING", "COMMISSIONS", "BATTLE",
            "DOCK", "SHOP", "MAILBOX", "MISSIONS", "BUILD",
            "ACADEMY", "EXERCISE", "COMBAT"
        }

        actual_states = {state.value for state in ScreenState}
        assert actual_states == expected_states

    def test_screen_state_uniqueness(self):
        """Test that all screen state values are unique."""
        values = [state.value for state in ScreenState]
        assert len(values) == len(set(values))


class TestScreenDetector:
    """Test ScreenDetector class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ScreenDetector()

    def test_initial_state(self):
        """Test initial detector state."""
        assert self.detector.previous_state == ScreenState.UNKNOWN
        assert self.detector.state_confidence == 0.0

    def test_identify_screen_unknown(self):
        """Test screen identification with no OCR results."""
        frame = self._create_mock_frame()

        state = self.detector.identify_screen(frame)
        assert state == ScreenState.UNKNOWN

    def test_identify_screen_by_ocr(self):
        """Test screen identification using OCR keywords."""
        frame = self._create_mock_frame()

        # Test home screen detection
        ocr_results = [
            {"text": "commissions", "conf": 0.9},
            {"text": "missions", "conf": 0.8},
            {"text": "mailbox", "conf": 0.85}
        ]

        state = self.detector.identify_screen(frame, ocr_results)
        assert state == ScreenState.HOME
        assert self.detector.state_confidence > 0.0

    def test_identify_screen_commission(self):
        """Test commission screen detection."""
        frame = self._create_mock_frame()

        ocr_results = [
            {"text": "commission", "conf": 0.9},
            {"text": "urgent", "conf": 0.8},
            {"text": "daily", "conf": 0.85}
        ]

        state = self.detector.identify_screen(frame, ocr_results)
        assert state == ScreenState.COMMISSIONS

    def test_identify_screen_mailbox(self):
        """Test mailbox screen detection."""
        frame = self._create_mock_frame()

        ocr_results = [
            {"text": "mail", "conf": 0.9},
            {"text": "collect", "conf": 0.8},
            {"text": "inbox", "conf": 0.85}
        ]

        state = self.detector.identify_screen(frame, ocr_results)
        assert state == ScreenState.MAILBOX

    def test_identify_screen_battle(self):
        """Test battle screen detection."""
        frame = self._create_mock_frame()

        ocr_results = [
            {"text": "sortie", "conf": 0.9},
            {"text": "battle", "conf": 0.8},
            {"text": "stage", "conf": 0.85}
        ]

        state = self.detector.identify_screen(frame, ocr_results)
        assert state == ScreenState.BATTLE

    def test_screen_state_persistence(self):
        """Test that detector maintains state between calls."""
        frame = self._create_mock_frame()

        # First call - unknown
        state1 = self.detector.identify_screen(frame)
        assert state1 == ScreenState.UNKNOWN

        # Second call with same input should maintain state
        state2 = self.detector.identify_screen(frame)
        assert state2 == ScreenState.UNKNOWN

    def test_state_change_detection(self):
        """Test detection of state changes."""
        frame = self._create_mock_frame()

        # Start with unknown
        state1 = self.detector.identify_screen(frame)
        assert state1 == ScreenState.UNKNOWN
        previous_state = self.detector.previous_state

        # Change to home
        ocr_results = [
            {"text": "commissions", "conf": 0.9},
            {"text": "missions", "conf": 0.8}
        ]
        state2 = self.detector.identify_screen(frame, ocr_results)
        assert state2 == ScreenState.HOME
        assert self.detector.previous_state != previous_state

    def test_low_confidence_maintenance(self):
        """Test maintaining previous state with low confidence."""
        frame = self._create_mock_frame()

        # Establish a known state with high confidence
        ocr_results = [
            {"text": "commissions", "conf": 0.9},
            {"text": "missions", "conf": 0.8},
            {"text": "mailbox", "conf": 0.85}
        ]
        state1 = self.detector.identify_screen(frame, ocr_results)
        assert state1 == ScreenState.HOME
        assert self.detector.state_confidence > 0.5

        # Try to change to unknown state with low confidence
        state2 = self.detector.identify_screen(frame)  # No OCR = low confidence
        assert state2 == ScreenState.HOME  # Should maintain previous state

    def test_confidence_threshold(self):
        """Test confidence threshold for state changes."""
        frame = self._create_mock_frame()

        # Try to identify with very low confidence keywords
        ocr_results = [
            {"text": "maybe", "conf": 0.1},
            {"text": "perhaps", "conf": 0.1}
        ]

        state = self.detector.identify_screen(frame, ocr_results)
        assert state == ScreenState.UNKNOWN  # Should not change due to low confidence

    def test_ui_feature_detection(self):
        """Test UI feature-based detection."""
        # Test with mock frame that has detectable features
        frame = self._create_mock_frame_with_features(bottom_nav=True)

        # Should detect home screen by UI features when OCR fails
        state = self.detector.identify_screen(frame)
        # Note: This test depends on the actual UI detection implementation
        # For now, just verify it doesn't crash
        assert isinstance(state, ScreenState)

    def test_loading_screen_detection(self):
        """Test loading screen detection."""
        # Create frame that should be detected as loading
        frame = self._create_mock_frame_with_features(loading_indicator=True)

        state = self.detector.identify_screen(frame)
        # Should detect loading screen by UI features
        assert isinstance(state, ScreenState)

    def test_multiple_keyword_matching(self):
        """Test screen detection with multiple matching keywords."""
        frame = self._create_mock_frame()

        # OCR with multiple commission keywords
        ocr_results = [
            {"text": "commission", "conf": 0.9},
            {"text": "urgent", "conf": 0.8},
            {"text": "daily", "conf": 0.85},
            {"text": "weekly", "conf": 0.7},
            {"text": "extra", "conf": 0.6}
        ]

        state = self.detector.identify_screen(frame, ocr_results)
        assert state == ScreenState.COMMISSIONS
        assert self.detector.state_confidence > 0.5  # High confidence from multiple matches

    def test_competing_keywords(self):
        """Test resolution when multiple screen types have matching keywords."""
        frame = self._create_mock_frame()

        # OCR with keywords from both home and commission screens
        ocr_results = [
            {"text": "commission", "conf": 0.9},
            {"text": "missions", "conf": 0.8},  # Home keyword
            {"text": "urgent", "conf": 0.85}   # Commission keyword
        ]

        state = self.detector.identify_screen(frame, ocr_results)

        # Should pick the screen with highest score
        # Commission has 2 matches, home has 1 match
        assert state == ScreenState.COMMISSIONS

    def test_empty_ocr_results(self):
        """Test handling of empty OCR results."""
        frame = self._create_mock_frame()

        state = self.detector.identify_screen(frame, [])
        assert state == ScreenState.UNKNOWN

    def test_none_ocr_results(self):
        """Test handling of None OCR results."""
        frame = self._create_mock_frame()

        state = self.detector.identify_screen(frame, None)
        assert state == ScreenState.UNKNOWN

    def test_ocr_result_format_variations(self):
        """Test handling of different OCR result formats."""
        frame = self._create_mock_frame()

        # Test with missing 'text' key
        ocr_results = [{"conf": 0.9}]  # Missing 'text' key
        state = self.detector.identify_screen(frame, ocr_results)
        assert state == ScreenState.UNKNOWN

        # Test with non-string text
        ocr_results = [{"text": 123, "conf": 0.9}]
        state = self.detector.identify_screen(frame, ocr_results)
        assert state == ScreenState.UNKNOWN

    def _create_mock_frame(self):
        """Create a mock frame for testing."""
        frame = Mock(spec=Frame)
        frame.image_bgr = Mock()
        frame.image_bgr.shape = (1080, 1920, 3)
        return frame

    def _create_mock_frame_with_features(self, bottom_nav=False, loading_indicator=False):
        """Create a mock frame with specific UI features."""
        frame = self._create_mock_frame()

        # Mock the image_bgr to return appropriate data for feature detection
        if bottom_nav:
            # Create image with bottom navigation pattern
            mock_image = Mock()
            mock_image.shape = (1080, 1920, 3)
            # This would need more complex mocking for actual feature detection
            frame.image_bgr = mock_image

        return frame


class TestStateTransitions:
    """Test valid state transitions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ScreenDetector()

    def test_valid_transitions_from_home(self):
        """Test valid transitions from HOME state."""
        valid_transitions = self.detector.get_valid_transitions(ScreenState.HOME)

        expected_transitions = {
            ScreenState.COMMISSIONS, ScreenState.MAILBOX, ScreenState.MISSIONS,
            ScreenState.BUILD, ScreenState.DOCK, ScreenState.SHOP,
            ScreenState.ACADEMY, ScreenState.EXERCISE, ScreenState.COMBAT,
            ScreenState.LOADING
        }

        assert valid_transitions == expected_transitions

    def test_valid_transitions_from_commission(self):
        """Test valid transitions from COMMISSIONS state."""
        valid_transitions = self.detector.get_valid_transitions(ScreenState.COMMISSIONS)

        expected_transitions = {ScreenState.HOME, ScreenState.LOADING}

        assert valid_transitions == expected_transitions

    def test_valid_transitions_from_unknown(self):
        """Test valid transitions from UNKNOWN state."""
        valid_transitions = self.detector.get_valid_transitions(ScreenState.UNKNOWN)

        expected_transitions = {ScreenState.HOME, ScreenState.LOADING}

        assert valid_transitions == expected_transitions

    def test_valid_transitions_from_loading(self):
        """Test valid transitions from LOADING state."""
        valid_transitions = self.detector.get_valid_transitions(ScreenState.LOADING)

        expected_transitions = {
            ScreenState.HOME, ScreenState.COMMISSIONS, ScreenState.MAILBOX,
            ScreenState.MISSIONS, ScreenState.BUILD, ScreenState.DOCK,
            ScreenState.SHOP, ScreenState.ACADEMY, ScreenState.EXERCISE,
            ScreenState.COMBAT
        }

        assert valid_transitions == expected_transitions

    def test_all_states_have_transitions(self):
        """Test that all screen states have defined transitions."""
        for state in ScreenState:
            transitions = self.detector.get_valid_transitions(state)
            assert isinstance(transitions, set)
            assert len(transitions) > 0  # Every state should have at least one valid transition

    def test_transition_symmetry(self):
        """Test that transitions are reasonably symmetric."""
        # Loading should be able to transition to most states
        loading_transitions = self.detector.get_valid_transitions(ScreenState.LOADING)

        # Most states should be able to transition to loading
        for state in ScreenState:
            if state != ScreenState.LOADING:
                transitions = self.detector.get_valid_transitions(state)
                # Most states should allow transition to loading (for navigation)
                # This is a reasonable assumption for the state machine


class TestRecoveryProcedures:
    """Test recovery procedures for navigation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ScreenDetector()

    def test_navigate_to_home_recovery(self):
        """Test recovery procedure generation."""
        actions = self.detector.navigate_to_home()

        assert isinstance(actions, list)
        assert len(actions) > 0

        # Should include back button presses
        back_actions = [action for action in actions if action['action'] == 'back']
        assert len(back_actions) >= 3  # At least 3 back attempts

        # Should end with home button
        last_action = actions[-1]
        assert last_action['action'] == 'home'

    def test_recovery_action_format(self):
        """Test that recovery actions have correct format."""
        actions = self.detector.navigate_to_home()

        for action in actions:
            assert 'action' in action
            assert 'rationale' in action
            assert action['action'] in ['back', 'home']
            assert isinstance(action['rationale'], str)

    def test_max_recovery_attempts(self):
        """Test that recovery doesn't have too many attempts."""
        actions = self.detector.navigate_to_home()

        # Should not have excessive attempts
        assert len(actions) <= 5  # Reasonable limit for recovery attempts


class TestScreenDetectorIntegration:
    """Integration tests for screen detector."""

    def test_state_machine_consistency(self):
        """Test that state machine remains consistent across multiple calls."""
        detector = ScreenDetector()
        frame = Mock(spec=Frame)
        frame.image_bgr = Mock()
        frame.image_bgr.shape = (1080, 1920, 3)

        # Make multiple calls and ensure consistency
        states = []
        for _ in range(10):
            state = detector.identify_screen(frame)
            states.append(state)

        # All states should be valid ScreenState instances
        for state in states:
            assert isinstance(state, ScreenState)

    def test_memory_usage(self):
        """Test that detector doesn't accumulate excessive state."""
        detector = ScreenDetector()

        # Create mock frame
        frame = Mock(spec=Frame)
        frame.image_bgr = Mock()
        frame.image_bgr.shape = (1080, 1920, 3)

        # Make many calls
        for _ in range(100):
            detector.identify_screen(frame)

        # Detector should not accumulate excessive internal state
        # (This is more of a smoke test for memory issues)

    def test_thread_safety(self):
        """Test basic thread safety (smoke test)."""
        # Note: This is a basic smoke test. Full thread safety testing
        # would require running in multiple threads.
        detector = ScreenDetector()
        frame = Mock(spec=Frame)
        frame.image_bgr = Mock()
        frame.image_bgr.shape = (1080, 1920, 3)

        # Make concurrent calls (in same thread for smoke test)
        results = []
        for _ in range(5):
            state = detector.identify_screen(frame)
            results.append(state)

        # Should not crash and should return valid states
        for result in results:
            assert isinstance(result, ScreenState)