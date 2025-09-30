"""LLM client and schemas for plan generation."""

import base64
import cv2
import json
import os
import re
import time
from typing import Any, Dict, Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field

from .capture import Frame
from .configs import LLMConfig


class Target(BaseModel):
    """Target selector for UI element."""
    kind: Literal["text", "icon", "bbox", "point", "region"]
    value: Optional[str] = None
    bbox: Optional[list[float]] = None  # [x,y,w,h] normalized
    point: Optional[list[float]] = None  # [x,y] normalized
    region_hint: Optional[str] = None  # e.g., "bottom_bar"
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class Step(BaseModel):
    """Single action step in a plan."""
    action: Literal["tap", "swipe", "wait", "back", "assert"]
    target: Optional[Target] = None
    rationale: Optional[str] = None


class Plan(BaseModel):
    """Complete action plan from LLM."""
    screen: str
    steps: list[Step]
    done: bool = False


class LLMClient:
    """Client for LLM-based plan generation."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize LLM client.

        Args:
            config: LLM configuration
        """
        self.config = config
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Gemini client."""
        try:
            import google.generativeai as genai

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")

            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel('gemini-1.5-flash-latest')

            logger.info("Gemini LLM client initialized successfully")

        except ImportError:
            logger.error("google-generativeai package not installed. Install with: uv add google-generativeai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise

    def propose_plan(self, frame: Frame, goal: Dict[str, Any], context: Dict[str, Any]) -> Plan:
        """Generate action plan for current frame and goal.

        Args:
            frame: Current screen frame
            goal: Goal description (e.g., {"action": "open_commissions"})
            context: Additional context (last screen, regions, etc.)

        Returns:
            Generated plan with steps
        """
        logger.debug(f"Requesting plan for goal: {goal}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Prepare prompt and image
                prompt = self._build_prompt(frame, goal, context)

                # Make API request with vision
                response_text = self._make_gemini_request(prompt, frame)

                # Parse JSON response
                plan = self._parse_plan(response_text)

                logger.info(f"Generated plan: {plan.screen}, {len(plan.steps)} steps, done={plan.done}")
                return plan

            except Exception as e:
                logger.warning(f"LLM request attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error("All LLM request attempts failed, returning fallback plan")
                    return self._fallback_plan("LLM service unavailable")

    def _build_prompt(self, frame: Frame, goal: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build comprehensive prompt for LLM.

        Args:
            frame: Current frame
            goal: Goal description
            context: Additional context

        Returns:
            Formatted prompt string
        """
        # Get device and viewport info
        device_w = context.get("device_w", frame.full_w)
        device_h = context.get("device_h", frame.full_h)
        active_rect = getattr(frame, 'active_rect', None)

        if active_rect:
            ax, ay, aw, ah = active_rect
            viewport_info = f"Active viewport: x={ax}, y={ay}, width={aw}, height={ah}"
        else:
            viewport_info = f"Full screen: width={device_w}, height={device_h}"

        # Get regions
        regions = context.get("regions", {})

        # Build comprehensive prompt
        prompt = f"""You are an expert UI automation agent for the mobile game Azur Lane.

GOAL: {json.dumps(goal, indent=2)}

DEVICE INFO:
- {viewport_info}
- All coordinates should be normalized [0.0-1.0] relative to the game viewport
- Origin (0.0, 0.0) = top-left of game area
- (1.0, 1.0) = bottom-right of game area
- (0.5, 0.5) = center of game area

SCREEN CONTEXT:
- Last screen: {context.get("last_screen", "unknown")}
- Available regions: {json.dumps(regions, indent=2)}

INSTRUCTIONS:
1. Analyze the screenshot to understand the current game state
2. Return ONLY a valid JSON Plan object with these exact requirements:
   - screen: current screen identifier (home, commissions, battle, dock, shop, loading, unknown)
   - steps: array of action steps (use minimal steps)
   - done: true if goal is already achieved, false otherwise

3. For each step, use these action types:
   - "tap": click/tap on element
   - "swipe": scroll or swipe gesture
   - "wait": pause for loading
   - "back": press back button
   - "assert": verify element exists

4. Target selectors (prefer in this order):
   - kind:"text" - for buttons with visible text
   - kind:"icon" - for buttons with icons
   - kind:"region" - for areas like "bottom_navigation", "dialog_center"
   - kind:"point" - only as last resort with [x,y] coordinates
   - kind:"bbox" - only as last resort with [x,y,w,h] coordinates

5. Always include:
   - region_hint when possible (e.g., "bottom_bar", "dialog", "navigation")
   - rationale explaining why this action achieves the goal
   - confidence score (0.0-1.0)

6. Coordinate system rules:
   - All coordinates are normalized [0.0-1.0]
   - (0.0, 0.0) = top-left of game viewport
   - (1.0, 1.0) = bottom-right of game viewport
   - Example: center button at (0.5, 0.8), top-right corner at (0.95, 0.05)

7. If uncertain about screen state, return single "back" step
8. If goal is already achieved, set done=true with empty steps array

Analyze the attached screenshot and provide your JSON response."""

        return prompt

    def _make_gemini_request(self, prompt: str, frame: Frame) -> str:
        """Make request to Gemini API with vision capabilities.

        Args:
            prompt: Text prompt
            frame: Frame containing screenshot

        Returns:
            Response text
        """
        if not self._client:
            raise RuntimeError("Gemini client not initialized")

        try:
            # Prepare image for Gemini
            import PIL.Image
            import io

            # Convert PNG bytes to PIL Image
            image = PIL.Image.open(io.BytesIO(frame.png_bytes))

            # Create content with both text and image
            response = self._client.generate_content(
                contents=[
                    prompt,
                    image
                ],
                generation_config={
                    "max_output_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "top_p": 0.8,
                    "top_k": 40
                }
            )

            # Extract response text
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    return candidate.content.parts[0].text

            raise ValueError("No valid response from Gemini API")

        except Exception as e:
            logger.error(f"Gemini API request failed: {e}")
            raise

    def _parse_plan(self, response_text: str) -> Plan:
        """Parse LLM response into Plan object.

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed Plan object
        """
        try:
            # Clean and extract JSON
            cleaned_text = self._clean_json_response(response_text)

            # Parse JSON
            plan_data = json.loads(cleaned_text)

            # Validate with Pydantic
            plan = Plan(**plan_data)

            # Additional validation
            self._validate_plan(plan)

            return plan

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response: {response_text}")

            # Return fallback plan
            return self._fallback_plan("Failed to parse LLM response")

    def _validate_plan(self, plan: Plan) -> None:
        """Validate plan structure and coordinates.

        Args:
            plan: Plan to validate

        Raises:
            ValueError: If plan is invalid
        """
        # Validate screen name
        valid_screens = {"home", "commissions", "battle", "dock", "shop", "loading", "unknown"}
        if plan.screen not in valid_screens:
            logger.warning(f"Unknown screen type: {plan.screen}")

        # Validate coordinates in steps
        for i, step in enumerate(plan.steps):
            if step.target:
                if step.target.point:
                    x, y = step.target.point
                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                        logger.warning(f"Step {i}: Invalid coordinates {step.target.point}")
                if step.target.bbox:
                    bx, by, bw, bh = step.target.bbox
                    if not all(0.0 <= coord <= 1.0 for coord in [bx, by, bw, bh]):
                        logger.warning(f"Step {i}: Invalid bbox {step.target.bbox}")

    def _clean_json_response(self, text: str) -> str:
        """Clean JSON response by removing markdown formatting.

        Args:
            text: Raw response text

        Returns:
            Cleaned JSON string
        """
        # Remove markdown code fences
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Extract JSON if wrapped in other text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        return text

    def _fallback_plan(self, reason: str) -> Plan:
        """Generate a safe fallback plan.

        Args:
            reason: Reason for fallback

        Returns:
            Safe fallback plan
        """
        return Plan(
            screen="unknown",
            steps=[Step(action="back", rationale=f"Fallback: {reason}")],
            done=False
        )

    def analyze_screen_with_vision(self, frame: Frame, prompt: str) -> dict:
        """Analyze screen using Gemini's vision capabilities."""
        if not self._client:
            return {"error": "LLM client not initialized"}
        
        try:
            # Prepare image for Gemini
            import PIL.Image
            import io

            # Convert PNG bytes to PIL Image
            image = PIL.Image.open(io.BytesIO(frame.png_bytes))

            # Create content with both text and image
            response = self._client.generate_content(
                contents=[
                    prompt,
                    image
                ],
                generation_config={
                    "max_output_tokens": 1024,
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "top_k": 40
                }
            )
            
            # Extract response text
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    response_text = candidate.content.parts[0].text
                    
                    # Try to parse as JSON
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError:
                        return {"response": response_text}
            
            return {"error": "No valid response from Gemini API"}
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return {"error": str(e)}