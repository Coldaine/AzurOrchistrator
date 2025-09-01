"""Unit tests for LLM client with mock API responses."""

import pytest
import json
from unittest.mock import Mock, patch, mock_open
from io import BytesIO
import PIL.Image

from azl_bot.core.llm_client import LLMClient, Target, Plan, Step
from azl_bot.core.capture import Frame
from azl_bot.core.configs import LLMConfig


class TestLLMClient:
    """Test LLM client functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = LLMConfig(
            provider="gemini",
            model="gemini-1.5-flash-latest",
            endpoint="https://generativelanguage.googleapis.com/v1beta",
            api_key_env="GEMINI_API_KEY",
            max_tokens=2048,
            temperature=0.1
        )

    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'})
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_client_initialization_success(self, mock_model, mock_configure):
        """Test successful client initialization."""
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance

        client = LLMClient(self.config)

        mock_configure.assert_called_once_with(api_key='test_key')
        mock_model.assert_called_once_with('gemini-1.5-flash-latest')
        assert client._client == mock_model_instance

    @patch.dict('os.environ', {}, clear=True)
    def test_client_initialization_missing_key(self):
        """Test client initialization failure with missing API key."""
        with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable not set"):
            LLMClient(self.config)

    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'})
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_client_initialization_import_error(self, mock_model, mock_configure):
        """Test client initialization with import error."""
        # Simulate import error
        with patch.dict('sys.modules', {'google.generativeai': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'google.generativeai'")):
                with pytest.raises(ImportError):
                    LLMClient(self.config)

    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'})
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_propose_plan_success(self, mock_model_class, mock_configure):
        """Test successful plan proposal."""
        # Mock the generative model
        mock_model = Mock()
        mock_response = Mock()
        mock_candidate = Mock()
        mock_content = Mock()
        mock_parts = [Mock()]

        mock_parts[0].text = '{"screen": "home", "steps": [{"action": "tap", "target": {"kind": "text", "value": "commissions"}}], "done": false}'
        mock_content.parts = mock_parts
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        mock_model.generate_content.return_value = mock_response

        mock_model_class.return_value = mock_model

        client = LLMClient(self.config)
        frame = self._create_mock_frame()
        goal = {"action": "open_commissions"}
        context = {"last_screen": "home"}

        plan = client.propose_plan(frame, goal, context)

        assert isinstance(plan, Plan)
        assert plan.screen == "home"
        assert len(plan.steps) == 1
        assert plan.steps[0].action == "tap"
        assert not plan.done

    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'})
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_propose_plan_with_image_processing(self, mock_model_class, mock_configure):
        """Test plan proposal with image processing."""
        mock_model = Mock()
        mock_response = Mock()
        mock_candidate = Mock()
        mock_content = Mock()
        mock_parts = [Mock()]

        mock_parts[0].text = '{"screen": "home", "steps": [], "done": true}'
        mock_content.parts = mock_parts
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        mock_model.generate_content.return_value = mock_response

        mock_model_class.return_value = mock_model

        client = LLMClient(self.config)
        frame = self._create_mock_frame()
        goal = {"action": "check_status"}
        context = {"last_screen": "home"}

        plan = client.propose_plan(frame, goal, context)

        # Verify that generate_content was called with both text and image
        call_args = mock_model.generate_content.call_args
        assert len(call_args[0]) == 2  # Should have text and image
        assert isinstance(call_args[0][1], PIL.Image.Image)  # Second argument should be PIL Image

    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'})
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_propose_plan_api_failure(self, mock_model_class, mock_configure):
        """Test plan proposal with API failure."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API Error")

        mock_model_class.return_value = mock_model

        client = LLMClient(self.config)
        frame = self._create_mock_frame()
        goal = {"action": "test"}
        context = {"last_screen": "home"}

        # Should return fallback plan after retries
        plan = client.propose_plan(frame, goal, context)

        assert isinstance(plan, Plan)
        assert plan.screen == "unknown"
        assert len(plan.steps) == 1
        assert plan.steps[0].action == "back"

    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'})
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_propose_plan_invalid_json(self, mock_model_class, mock_configure):
        """Test plan proposal with invalid JSON response."""
        mock_model = Mock()
        mock_response = Mock()
        mock_candidate = Mock()
        mock_content = Mock()
        mock_parts = [Mock()]

        mock_parts[0].text = "Invalid JSON response"
        mock_content.parts = mock_parts
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        mock_model.generate_content.return_value = mock_response

        mock_model_class.return_value = mock_model

        client = LLMClient(self.config)
        frame = self._create_mock_frame()
        goal = {"action": "test"}
        context = {"last_screen": "home"}

        plan = client.propose_plan(frame, goal, context)

        # Should return fallback plan
        assert isinstance(plan, Plan)
        assert plan.screen == "unknown"

    def test_build_prompt_comprehensive(self):
        """Test comprehensive prompt building."""
        client = LLMClient(self.config)
        frame = self._create_mock_frame()
        goal = {"action": "open_commissions", "target": "urgent"}
        context = {
            "device_w": 1920,
            "device_h": 1080,
            "regions": {"top_bar": [0, 0, 1920, 162]},
            "last_screen": "home"
        }

        prompt = client._build_prompt(frame, goal, context)

        assert isinstance(prompt, str)
        assert "GOAL:" in prompt
        assert "DEVICE INFO:" in prompt
        assert "SCREEN CONTEXT:" in prompt
        assert "INSTRUCTIONS:" in prompt
        assert "open_commissions" in prompt
        assert "home" in prompt

    def test_build_prompt_with_active_area(self):
        """Test prompt building with active area information."""
        client = LLMClient(self.config)
        frame = Frame(
            png_bytes=b"test",
            image_bgr=Mock(),
            full_w=1920,
            full_h=1080,
            active_rect=(60, 132, 1800, 816),
            ts=0.0
        )
        goal = {"action": "test"}
        context = {"device_w": 1920, "device_h": 1080}

        prompt = client._build_prompt(frame, goal, context)

        assert "Active viewport:" in prompt
        assert "x=60, y=132" in prompt
        assert "width=1800, height=816" in prompt

    def test_parse_plan_valid(self):
        """Test parsing valid plan JSON."""
        client = LLMClient(self.config)

        json_response = '''{
            "screen": "commissions",
            "steps": [
                {
                    "action": "tap",
                    "target": {
                        "kind": "text",
                        "value": "urgent",
                        "confidence": 0.9
                    },
                    "rationale": "Click urgent commission"
                }
            ],
            "done": false
        }'''

        plan = client._parse_plan(json_response)

        assert isinstance(plan, Plan)
        assert plan.screen == "commissions"
        assert len(plan.steps) == 1
        assert plan.steps[0].action == "tap"
        assert not plan.done

    def test_parse_plan_with_markdown(self):
        """Test parsing plan JSON wrapped in markdown."""
        client = LLMClient(self.config)

        json_response = '''```json
        {
            "screen": "home",
            "steps": [],
            "done": true
        }
        ```'''

        plan = client._parse_plan(json_response)

        assert isinstance(plan, Plan)
        assert plan.screen == "home"
        assert plan.done

    def test_parse_plan_invalid_json(self):
        """Test parsing invalid JSON."""
        client = LLMClient(self.config)

        invalid_response = "This is not JSON"

        plan = client._parse_plan(invalid_response)

        # Should return fallback plan
        assert isinstance(plan, Plan)
        assert plan.screen == "unknown"
        assert len(plan.steps) == 1
        assert plan.steps[0].action == "back"

    def test_parse_plan_missing_fields(self):
        """Test parsing JSON with missing required fields."""
        client = LLMClient(self.config)

        incomplete_json = '{"screen": "home"}'

        plan = client._parse_plan(incomplete_json)

        # Should return fallback plan
        assert isinstance(plan, Plan)
        assert plan.screen == "unknown"

    def test_clean_json_response_various_formats(self):
        """Test cleaning various JSON response formats."""
        client = LLMClient(self.config)

        # Test markdown with json
        markdown_response = '''```json
        {"screen": "home", "steps": [], "done": true}
        ```'''
        cleaned = client._clean_json_response(markdown_response)
        assert '"screen": "home"' in cleaned

        # Test plain JSON
        plain_json = '{"screen": "home"}'
        cleaned = client._clean_json_response(plain_json)
        assert cleaned == plain_json

        # Test JSON with extra text
        mixed_response = '''Some text before
        {"screen": "home"}
        Some text after'''
        cleaned = client._clean_json_response(mixed_response)
        assert '"screen": "home"' in cleaned

    def test_validate_plan_coordinates(self):
        """Test plan validation with coordinate checking."""
        client = LLMClient(self.config)

        # Valid plan
        valid_plan = Plan(
            screen="home",
            steps=[
                Step(
                    action="tap",
                    target=Target(kind="point", point=[0.5, 0.5])
                )
            ]
        )

        # Should not raise exception
        client._validate_plan(valid_plan)

        # Invalid plan with out-of-bounds coordinates
        invalid_plan = Plan(
            screen="home",
            steps=[
                Step(
                    action="tap",
                    target=Target(kind="point", point=[1.5, 0.5])  # X > 1.0
                )
            ]
        )

        # Should not raise exception but log warning (validation is informational)
        client._validate_plan(invalid_plan)

    def test_fallback_plan_generation(self):
        """Test fallback plan generation."""
        client = LLMClient(self.config)

        fallback_plan = client._fallback_plan("Test reason")

        assert isinstance(fallback_plan, Plan)
        assert fallback_plan.screen == "unknown"
        assert len(fallback_plan.steps) == 1
        assert fallback_plan.steps[0].action == "back"
        assert "Test reason" in fallback_plan.steps[0].rationale
        assert not fallback_plan.done

    def test_target_model_validation(self):
        """Test Target model validation."""
        # Valid target
        target = Target(kind="text", value="commissions", confidence=0.8)
        assert target.kind == "text"
        assert target.value == "commissions"

        # Invalid confidence
        with pytest.raises(ValueError):
            Target(kind="text", confidence=1.5)  # > 1.0

        with pytest.raises(ValueError):
            Target(kind="text", confidence=-0.1)  # < 0.0

    def test_step_model_validation(self):
        """Test Step model validation."""
        # Valid step
        step = Step(action="tap", target=Target(kind="text", value="test"))
        assert step.action == "tap"
        assert step.target is not None

        # Invalid action
        with pytest.raises(ValueError):
            Step(action="invalid_action")

    def test_plan_model_validation(self):
        """Test Plan model validation."""
        # Valid plan
        plan = Plan(screen="home", steps=[], done=True)
        assert plan.screen == "home"
        assert plan.done

        # Invalid screen
        plan = Plan(screen="invalid_screen", steps=[], done=False)
        # Should still create but screen is non-standard
        assert plan.screen == "invalid_screen"

    def _create_mock_frame(self):
        """Create a mock frame for testing."""
        frame = Mock(spec=Frame)
        frame.png_bytes = b"fake_png_data"
        frame.image_bgr = Mock()
        frame.image_bgr.shape = (1080, 1920, 3)
        frame.full_w = 1920
        frame.full_h = 1080
        frame.active_rect = None
        frame.ts = 0.0
        return frame

    def _create_real_frame(self):
        """Create a real frame with actual image data for testing."""
        # Create a small test image
        import numpy as np
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)

        frame = Frame(
            png_bytes=b"test",
            image_bgr=test_image,
            full_w=1920,
            full_h=1080,
            active_rect=(0, 0, 1920, 1080),
            ts=0.0
        )
        return frame


class TestLLMIntegration:
    """Integration tests for LLM functionality."""

    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'})
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_full_pipeline(self, mock_model_class, mock_configure):
        """Test full LLM pipeline from frame to plan."""
        # Mock successful API response
        mock_model = Mock()
        mock_response = Mock()
        mock_candidate = Mock()
        mock_content = Mock()
        mock_parts = [Mock()]

        mock_parts[0].text = '''{
            "screen": "commissions",
            "steps": [
                {
                    "action": "tap",
                    "target": {
                        "kind": "text",
                        "value": "urgent commission",
                        "confidence": 0.95
                    },
                    "rationale": "Tap on urgent commission to collect rewards"
                }
            ],
            "done": false
        }'''

        mock_content.parts = mock_parts
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        mock_model.generate_content.return_value = mock_response

        mock_model_class.return_value = mock_model

        client = LLMClient(self.config)

        # Create test frame
        frame = Frame(
            png_bytes=b"test_image",
            image_bgr=Mock(),
            full_w=1920,
            full_h=1080,
            active_rect=(60, 132, 1800, 816),
            ts=0.0
        )

        goal = {"action": "collect_commissions"}
        context = {
            "device_w": 1920,
            "device_h": 1080,
            "last_screen": "home",
            "regions": {"center": [0.2, 0.12, 0.6, 0.73]}
        }

        plan = client.propose_plan(frame, goal, context)

        assert plan.screen == "commissions"
        assert len(plan.steps) == 1
        assert plan.steps[0].action == "tap"
        assert plan.steps[0].target.value == "urgent commission"
        assert not plan.done

        # Verify API was called with correct parameters
        call_args = mock_model.generate_content.call_args
        assert len(call_args[0]) == 2  # Text prompt and image

    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'})
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_error_recovery(self, mock_model_class, mock_configure):
        """Test error recovery and retry logic."""
        mock_model = Mock()

        # First call fails, second succeeds
        mock_model.generate_content.side_effect = [
            Exception("Network error"),
            Mock(
                candidates=[
                    Mock(
                        content=Mock(
                            parts=[Mock(text='{"screen": "home", "steps": [], "done": true}')]
                        )
                    )
                ]
            )
        ]

        mock_model_class.return_value = mock_model

        client = LLMClient(self.config)
        frame = self._create_mock_frame()
        goal = {"action": "test"}
        context = {"last_screen": "home"}

        plan = client.propose_plan(frame, goal, context)

        # Should eventually succeed
        assert plan.screen == "home"
        assert plan.done

        # Should have been called twice (initial + retry)
        assert mock_model.generate_content.call_count == 2

    def _create_mock_frame(self):
        """Create a mock frame for testing."""
        frame = Mock(spec=Frame)
        frame.png_bytes = b"fake_png_data"
        frame.image_bgr = Mock()
        frame.image_bgr.shape = (1080, 1920, 3)
        frame.full_w = 1920
        frame.full_h = 1080
        frame.active_rect = None
        frame.ts = 0.0
        return frame