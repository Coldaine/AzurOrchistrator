
import json
import re
from typing import List, Optional, Tuple, Dict, Any

from loguru import logger

from .capture import Frame
from .llm_client import LLMClient, Target


class Candidate:
    """Detection candidate with metadata."""
    
    def __init__(self, point: Tuple[float, float], confidence: float, method: str):
        """Initialize candidate.
        
        Args:
            point: (x, y) normalized coordinates
            confidence: Detection confidence (0.0-1.0)
            method: Detection method name
        """
        self.point = point
        self.confidence = confidence
        self.method = method


class Resolver:
    """Resolves UI element locations using multiple detection methods."""
    
    def __init__(self, config: Dict[str, Any], ocr_client, templates_dir: str, llm: Optional[LLMClient] = None):
        """Initialize resolver.
        
        Args:
            config: Resolver configuration
            ocr_client: OCR client instance
            templates_dir: Path to template directory
            llm: Optional LLM client for arbitration
        """
        self.config = config
        self.ocr = ocr_client
        self.templates_dir = templates_dir
        self.llm = llm
        self._llm_cache: Dict[str, Candidate] = {}
        
    def resolve(self, target: Target, frame: Frame) -> Optional[Candidate]:
        """Resolve target location using multiple detection methods.
        
        Args:
            target: Target to resolve
            frame: Current screen frame
            
        Returns:
            Best candidate or None if not found
        """
        candidates = []
        
        # Try different detection methods based on target kind
        if target.kind == "text":
            candidates.extend(self._detect_by_text(target, frame))
        elif target.kind == "icon":
            candidates.extend(self._detect_by_template(target, frame))
        elif target.kind == "region":
            candidates.extend(self._detect_by_region(target, frame))
        elif target.kind == "point":
            # Direct point specification
            if target.point:
                candidates.append(Candidate(target.point, target.confidence, "direct"))
        
        if not candidates:
            return None
            
        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        # Check if top candidates disagree significantly
        if len(candidates) > 1 and max(c.confidence for c in candidates) < 0.8:
            # Methods disagree and no high confidence - use LLM
            llm_choice = self._arbitrate_with_llm(target, frame, candidates)
            if llm_choice:
                return llm_choice
        
        return candidates[0]
    
    def _detect_by_text(self, target: Target, frame: Frame) -> List[Candidate]:
        """Detect by OCR text matching."""
        candidates = []
        
        # Use OCR to find text
        ocr_results = self.ocr.extract_text(frame)
        
        for result in ocr_results:
            if target.value.lower() in result['text'].lower():
                # Convert OCR bbox to normalized coordinates
                bbox = result['bbox']  # [x1, y1, x2, y2] in pixels
                center_x = (bbox[0] + bbox[2]) / 2 / frame.full_w
                center_y = (bbox[1] + bbox[3]) / 2 / frame.full_h
                
                confidence = result.get('confidence', 0.7)
                candidates.append(Candidate((center_x, center_y), confidence, "ocr"))
        
        return candidates
    
    def _detect_by_template(self, target: Target, frame: Frame) -> List[Candidate]:
        """Detect by template matching."""
        # Simplified template matching
        # In a full implementation, this would load and match templates
        return []
    
    def _detect_by_region(self, target: Target, frame: Frame) -> List[Candidate]:
        """Detect by region hints."""
        # Use region hints to estimate location
        if target.region_hint == "center":
            return [Candidate((0.5, 0.5), 0.5, "region_hint")]
        elif target.region_hint == "bottom_bar":
            return [Candidate((0.5, 0.9), 0.6, "region_hint")]
        
        return []
    
    def _arbitrate_with_llm(self, target: Target, frame: Frame, candidates: List[Candidate]) -> Optional[Candidate]:
        """Use LLM vision to arbitrate between disagreeing methods."""
        if not self.llm or not hasattr(self.llm, 'analyze_screen_with_vision'):
            return None
        
        # Check if candidates actually disagree significantly
        if not self._methods_disagree(candidates):
            return None
        
        logger.info(f"Methods disagree on {target.kind}='{target.value}', using LLM arbitration")
        
        try:
            # Create arbitration prompt
            arbitration_prompt = self._build_arbitration_prompt(target, candidates)
            
            # Use LLM vision analysis
            result = self.llm.analyze_screen_with_vision(frame, arbitration_prompt)
            
            if isinstance(result, dict) and 'response' in result:
                # Parse coordinates from response
                point = self._parse_llm_coordinates(result['response'])
                if point:
                    candidate = Candidate(point, 0.9, "llm_arbitration")  # High confidence for LLM
                    return candidate
            
        except Exception as e:
            logger.warning(f"LLM arbitration failed: {e}")
        
        return None
    
    def _methods_disagree(self, candidates: List[Candidate], threshold: float = 0.05) -> bool:
        """Check if methods disagree significantly."""
        if len(candidates) < 2:
            return False
        
        # Calculate pairwise distances
        points = [c.point for c in candidates]
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = ((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)**0.5
                if dist > threshold:
                    return True
        
        return False
    
    def _build_arbitration_prompt(self, target: Target, candidates: List[Candidate]) -> str:
        """Build prompt for LLM arbitration."""
        prompt = f"""I need help resolving a disagreement between different computer vision methods trying to locate: "{target.value}"

The methods found these locations (normalized coordinates 0.0-1.0):
"""

        for i, candidate in enumerate(candidates):
            x, y = candidate.point
            prompt += f"{i+1}. {candidate.method}: ({x:.3f}, {y:.3f}) confidence: {candidate.confidence:.2f}\n"

        prompt += f"""
Please analyze the screenshot and tell me the most accurate location for "{target.value}".

Consider:
- Which location makes the most visual sense
- The context and layout of the UI elements
- The type of element we're looking for ({target.kind})

Return your answer as normalized coordinates in the format: (x.xxx, y.yyy)
Where x and y are between 0.0 and 1.0, with (0,0) being top-left and (1,1) being bottom-right.

Your response should contain ONLY the coordinate pair, nothing else."""

        return prompt
    
    def _parse_llm_coordinates(self, response: str) -> Optional[Tuple[float, float]]:
        """Parse coordinates from LLM response."""
        # Look for coordinate pattern like (0.123, 0.456)
        coord_pattern = r'\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)'
        match = re.search(coord_pattern, response)
        
        if match:
            try:
                x = float(match.group(1))
                y = float(match.group(2))
                
                # Validate coordinates
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    return (x, y)
                else:
                    logger.warning(f"LLM returned invalid coordinates: ({x}, {y})")
            except ValueError as e:
                logger.warning(f"Failed to parse LLM coordinates: {e}")
        
        logger.warning(f"Could not parse coordinates from LLM response: {response}")
        return None