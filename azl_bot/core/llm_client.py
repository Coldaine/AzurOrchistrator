"""LLM client and schemas for plan generation."""

import base64
import json
import re
from typing import Any, Dict, Literal, Optional

import requests
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
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "AzurLaneBot/0.1.0"
        })
    
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
        
        # Prepare prompt
        prompt = self._build_prompt(frame, goal, context)
        
        # Make API request
        response_text = self._make_request(prompt)
        
        # Parse JSON response
        plan = self._parse_plan(response_text)
        
        logger.info(f"Generated plan: {plan.screen}, {len(plan.steps)} steps, done={plan.done}")
        return plan
    
    def _build_prompt(self, frame: Frame, goal: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build prompt for LLM.
        
        Args:
            frame: Current frame
            goal: Goal description
            context: Additional context
            
        Returns:
            Formatted prompt string
        """
        # Encode frame as base64
        frame_b64 = base64.b64encode(frame.png_bytes).decode('utf-8')
        
        # Get device info
        device_w = context.get("device_w", frame.full_w)
        device_h = context.get("device_h", frame.full_h)
        
        # Get regions
        regions = context.get("regions", {})
        
        # Build prompt
        prompt = f"""GOAL:
{json.dumps(goal)}

DEVICE:
width={device_w}, height={device_h}, regions={json.dumps(regions)}

LAST_SCREEN:
{context.get("last_screen", "unknown")}

INSTRUCTIONS:
Return a Plan with minimal steps to achieve the goal. Use kind:"text" when a button has a label. Always include a region_hint if possible. If the goal is already achieved, set done=true.

CURRENT_FRAME_BASE64:
{frame_b64}"""
        
        return prompt
    
    def _make_request(self, prompt: str) -> str:
        """Make API request to LLM.
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            LLM response text
        """
        # System prompt
        system_prompt = ("You are a UI navigator for the mobile game Azur Lane running on an Android emulator. "
                        "Respond only with JSON conforming to the schema provided. "
                        "Prefer text/icon selectors with region_hint over coordinates. "
                        "If uncertain, return a single back step. Be concise.")
        
        # Prepare request based on provider
        if self.config.provider == "gemini":
            response_text = self._make_gemini_request(system_prompt, prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
            
        return response_text
    
    def _make_gemini_request(self, system_prompt: str, user_prompt: str) -> str:
        """Make request to Gemini API.
        
        Args:
            system_prompt: System instruction
            user_prompt: User prompt with image
            
        Returns:
            Response text
        """
        import os
        
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment: {self.config.api_key_env}")
        
        # Build request data
        url = f"{self.config.endpoint}/models/{self.config.model}:generateContent"
        
        # For this implementation, we'll use a simplified text-only approach
        # In a full implementation, you'd handle image inputs properly
        data = {
            "contents": [{
                "parts": [{
                    "text": f"{system_prompt}\n\n{user_prompt}"
                }]
            }],
            "generationConfig": {
                "maxOutputTokens": self.config.max_tokens,
                "temperature": self.config.temperature
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
        
        try:
            response = self._session.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract text from response
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if len(parts) > 0 and "text" in parts[0]:
                        return parts[0]["text"]
            
            raise ValueError("Unexpected response format from Gemini API")
            
        except requests.RequestException as e:
            logger.error(f"LLM API request failed: {e}")
            # Return a fallback plan
            return '{"screen": "unknown", "steps": [{"action": "back", "rationale": "API error, returning to safe state"}], "done": false}'
    
    def _parse_plan(self, response_text: str) -> Plan:
        """Parse LLM response into Plan object.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Parsed Plan object
        """
        try:
            # Strip markdown code fences if present
            cleaned_text = self._clean_json_response(response_text)
            
            # Parse JSON
            plan_data = json.loads(cleaned_text)
            
            # Validate with Pydantic
            plan = Plan(**plan_data)
            
            return plan
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response: {response_text}")
            
            # Return fallback plan
            return Plan(
                screen="unknown",
                steps=[Step(action="back", rationale="Failed to parse LLM response")],
                done=False
            )
    
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
        
        return text