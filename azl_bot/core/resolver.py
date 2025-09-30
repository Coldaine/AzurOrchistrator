zl_bot/core/resolver.py</path>
<content">"""UI element resolution and detection."""

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
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
        self.templates_dir = Path(templates_dir)
        self.llm = llm
        self._llm_cache: Dict[str, Candidate] = {}
        self._template_cache: Dict[str, List[Tuple[np.ndarray, float]]] = {}
        self._template_edge_cache: Dict[str, List[Tuple[np.ndarray, float]]] = {}
        
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
            logger.debug(f"No candidates found for {target.kind}:{target.value}")
            return None
        
        # Apply confidence weighting
        thresholds = self.config.get("thresholds", {})
        weights = thresholds.get("weights", {})
        
        for candidate in candidates:
            weight = weights.get(candidate.method, 1.0)
            candidate.confidence = min(1.0, candidate.confidence * weight)
        
        # Sort by weighted confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        # Log all candidates
        logger.info(f"Candidates for {target.kind}:{target.value}:")
        for i, c in enumerate(candidates):
            logger.info(f"  {i+1}. {c.method}: ({c.point[0]:.3f}, {c.point[1]:.3f}) conf={c.confidence:.3f}")
        
        # Check if top candidates disagree significantly
        if len(candidates) > 1 and max(c.confidence for c in candidates) < 0.8:
            # Methods disagree and no high confidence - use LLM
            llm_choice = self._arbitrate_with_llm(target, frame, candidates)
            if llm_choice:
                logger.info(f"LLM arbitration selected: ({llm_choice.point[0]:.3f}, {llm_choice.point[1]:.3f})")
                return llm_choice
        
        logger.info(f"Selected: {candidates[0].method} ({candidates[0].point[0]:.3f}, {candidates[0].point[1]:.3f}) conf={candidates[0].confidence:.3f}")
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
        """Detect by template matching with multi-scale edge-NCC and ORB fallback.
        
        Args:
            target: Target with icon/template name
            frame: Current frame
            
        Returns:
            List of candidates from template matching
        """
        candidates = []
        
        # Load template
        template_name = target.value
        template_path = self.templates_dir / f"{template_name}.png"
        
        if not template_path.exists():
            logger.debug(f"Template not found: {template_path}")
            return candidates
        
        # Load and cache template pyramid
        if template_name not in self._template_cache:
            template_bgr = cv2.imread(str(template_path))
            if template_bgr is None:
                logger.warning(f"Failed to load template: {template_path}")
                return candidates
            
            # Build multi-scale pyramid (scales: 1.0, 0.9, 0.8, 1.1, 1.2)
            scales = [1.0, 0.9, 0.8, 1.1, 1.2]
            pyramid = []
            edge_pyramid = []
            
            for scale in scales:
                if scale != 1.0:
                    h, w = template_bgr.shape[:2]
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    scaled = cv2.resize(template_bgr, (new_w, new_h))
                else:
                    scaled = template_bgr
                
                pyramid.append((scaled, scale))
                
                # Compute edges for this scale
                gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_pyramid.append((edges, scale))
            
            self._template_cache[template_name] = pyramid
            self._template_edge_cache[template_name] = edge_pyramid
            logger.debug(f"Cached template pyramid for {template_name}")
        
        # Get cached pyramids
        pyramid = self._template_cache[template_name]
        edge_pyramid = self._template_edge_cache[template_name]
        
        # Prepare frame
        frame_gray = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2GRAY)
        frame_edges = cv2.Canny(frame_gray, 50, 150)
        
        # Get thresholds
        thresholds = self.config.get("thresholds", {})
        ncc_edge_thresh = thresholds.get("ncc_edge", 0.60)
        ncc_gray_thresh = thresholds.get("ncc_gray", 0.70)
        orb_inliers_thresh = thresholds.get("orb_inliers", 12)
        
        best_ncc_score = 0.0
        best_ncc_location = None
        best_scale = 1.0
        
        # Try edge-based NCC at multiple scales
        for edges, scale in edge_pyramid:
            if edges.shape[0] > frame_edges.shape[0] or edges.shape[1] > frame_edges.shape[1]:
                continue
            
            result = cv2.matchTemplate(frame_edges, edges, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_ncc_score:
                best_ncc_score = max_val
                best_ncc_location = max_loc
                best_scale = scale
        
        # If edge NCC is above threshold, add candidate
        if best_ncc_score >= ncc_edge_thresh and best_ncc_location:
            # Get template size at best scale
            for tmpl, scale in pyramid:
                if scale == best_scale:
                    tmpl_h, tmpl_w = tmpl.shape[:2]
                    break
            
            # Calculate center point
            center_x = best_ncc_location[0] + tmpl_w // 2
            center_y = best_ncc_location[1] + tmpl_h // 2
            
            # Convert to normalized coordinates
            x_norm = center_x / frame.image_bgr.shape[1]
            y_norm = center_y / frame.image_bgr.shape[0]
            
            confidence = float(best_ncc_score)
            candidates.append(Candidate((x_norm, y_norm), confidence, "template"))
            logger.debug(f"Template NCC match: {confidence:.3f} at ({x_norm:.3f}, {y_norm:.3f})")
        
        # If NCC confidence is low, try ORB fallback
        if best_ncc_score < ncc_gray_thresh:
            logger.debug(f"NCC score {best_ncc_score:.3f} below threshold, trying ORB fallback")
            orb_candidate = self._detect_by_orb(target, frame, pyramid[0][0])
            
            if orb_candidate:
                candidates.append(orb_candidate)
        
        return candidates
    
    def _detect_by_orb(self, target: Target, frame: Frame, template: np.ndarray) -> Optional[Candidate]:
        """Detect by ORB feature matching.
        
        Args:
            target: Target specification
            frame: Current frame
            template: Template image
            
        Returns:
            Candidate if enough inliers found, else None
        """
        # Get threshold
        thresholds = self.config.get("thresholds", {})
        orb_inliers_thresh = thresholds.get("orb_inliers", 12)
        
        # Initialize ORB
        orb = cv2.ORB_create(nfeatures=1500)
        
        # Convert to grayscale
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect and compute
        kp1, des1 = orb.detectAndCompute(template_gray, None)
        kp2, des2 = orb.detectAndCompute(frame_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            logger.debug("ORB: Insufficient keypoints")
            return None
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        logger.debug(f"ORB: {len(good_matches)} good matches")
        
        if len(good_matches) < orb_inliers_thresh:
            return None
        
        # Get matched keypoint locations
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        if len(good_matches) >= 4:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is not None and mask is not None:
                inliers = mask.ravel().sum()
                logger.debug(f"ORB: {inliers} inliers")
                
                if inliers >= orb_inliers_thresh:
                    # Get center of template in frame
                    h, w = template.shape[:2]
                    template_center = np.float32([[w/2, h/2, 1]]).T
                    frame_center = H @ template_center
                    
                    if frame_center[2, 0] != 0:
                        center_x = frame_center[0, 0] / frame_center[2, 0]
                        center_y = frame_center[1, 0] / frame_center[2, 0]
                        
                        # Convert to normalized
                        x_norm = center_x / frame.image_bgr.shape[1]
                        y_norm = center_y / frame.image_bgr.shape[0]
                        
                        # Confidence based on inlier ratio
                        confidence = min(1.0, inliers / (len(good_matches) + 1))
                        
                        logger.debug(f"ORB match: {confidence:.3f} at ({x_norm:.3f}, {y_norm:.3f})")
                        return Candidate((x_norm, y_norm), confidence, "orb")
        
        return None
    
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