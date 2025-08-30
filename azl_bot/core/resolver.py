"""Selector resolver for converting LLM targets to pixel coordinates."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml
from loguru import logger
from rapidfuzz import fuzz

from .capture import Frame
from .configs import ResolverConfig
from .llm_client import Target
from .ocr import OCRClient


@dataclass
class Candidate:
    """Candidate location for a target selector."""
    point: Tuple[float, float]  # normalized (active area)
    confidence: float
    method: str  # "ocr", "template", "orb", "llm"


class Resolver:
    """Resolves LLM selectors to pixel coordinates using multiple methods."""
    
    def __init__(self, config: ResolverConfig, ocr: OCRClient, templates_dir: Path) -> None:
        """Initialize resolver.
        
        Args:
            config: Resolver configuration
            ocr: OCR client instance
            templates_dir: Directory containing template images
        """
        self.config = config
        self.ocr = ocr
        self.templates_dir = Path(templates_dir)
        self._templates = {}
        self._synonyms = {}
        self._load_templates()
        self._load_synonyms()
    
    def _load_templates(self) -> None:
        """Load template images for matching."""
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return
        
        for template_path in self.templates_dir.glob("*.png"):
            name = template_path.stem
            try:
                template_bgr = cv2.imread(str(template_path))
                if template_bgr is not None:
                    # Precompute edge map for template matching
                    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
                    template_edges = cv2.Canny(template_gray, 50, 150)
                    
                    self._templates[name] = {
                        'bgr': template_bgr,
                        'gray': template_gray,
                        'edges': template_edges
                    }
                    logger.debug(f"Loaded template: {name}")
            except Exception as e:
                logger.warning(f"Failed to load template {template_path}: {e}")
    
    def _load_synonyms(self) -> None:
        """Load synonym mappings from YAML."""
        synonyms_path = self.templates_dir.parent / "selectors" / "synonyms.yaml"
        
        if synonyms_path.exists():
            try:
                with open(synonyms_path, 'r', encoding='utf-8') as f:
                    self._synonyms = yaml.safe_load(f) or {}
                logger.debug(f"Loaded {len(self._synonyms)} synonym groups")
            except Exception as e:
                logger.warning(f"Failed to load synonyms: {e}")
    
    def resolve(self, selector: Target, frame: Frame) -> Optional[Candidate]:
        """Resolve a target selector to a candidate location.
        
        Args:
            selector: Target selector from LLM
            frame: Current screen frame
            
        Returns:
            Best candidate location, or None if not found
        """
        logger.debug(f"Resolving selector: {selector.kind}={selector.value} in {selector.region_hint}")
        
        candidates = []
        
        # Try different resolution methods
        if selector.kind == "text" and selector.value:
            candidates.extend(self._resolve_text(selector, frame))
        
        if selector.kind == "icon" and selector.value:
            candidates.extend(self._resolve_icon(selector, frame))
        
        if selector.kind == "bbox" and selector.bbox:
            candidates.append(self._resolve_bbox(selector, frame))
        
        if selector.kind == "point" and selector.point:
            candidates.append(self._resolve_point(selector, frame))
        
        if selector.kind == "region" and selector.value:
            candidates.append(self._resolve_region(selector, frame))
        
        # Filter and rank candidates
        valid_candidates = [c for c in candidates if c and self._is_valid_point(c.point)]
        
        if not valid_candidates:
            logger.warning(f"No valid candidates found for {selector}")
            return None
        
        # Apply confidence fusion for agreeing methods
        fused_candidates = self._fuse_candidates(valid_candidates)
        
        # Select best candidate
        best = max(fused_candidates, key=lambda c: c.confidence)
        
        # Apply confidence gating
        if best.confidence < self.config.thresholds.combo_accept:
            logger.warning(f"Best candidate confidence {best.confidence:.3f} below threshold {self.config.thresholds.combo_accept}")
            return None
        
        logger.info(f"Resolved {selector.kind}='{selector.value}' to {best.point} via {best.method} (conf={best.confidence:.3f})")
        return best
    
    def _resolve_text(self, selector: Target, frame: Frame) -> List[Candidate]:
        """Resolve text selector using OCR."""
        candidates = []
        
        # Determine ROI
        roi = self._get_roi_for_region(selector.region_hint)
        
        # Run OCR
        ocr_results = self.ocr.text_in_roi(frame.image_bgr, roi)
        
        # Find matching text
        for result in ocr_results:
            text = result['text']
            conf = result['conf']
            
            # Check if text matches using synonyms and fuzzy matching
            match_score = self._text_match_score(selector.value, text)
            
            if match_score >= self.config.thresholds.ocr_text:
                # Calculate center point of text box
                box = result['box_norm']
                center_x = box[0] + box[2] / 2
                center_y = box[1] + box[3] / 2
                
                # Convert from full image to active area coordinates
                point = self._convert_full_to_active(center_x, center_y, frame)
                
                combined_conf = (match_score + conf) / 2
                candidates.append(Candidate(point, combined_conf, "ocr"))
        
        return candidates
    
    def _resolve_icon(self, selector: Target, frame: Frame) -> List[Candidate]:
        """Resolve icon selector using template matching."""
        candidates = []
        
        icon_name = selector.value
        if icon_name not in self._templates:
            logger.warning(f"Template not found for icon: {icon_name}")
            return candidates
        
        template = self._templates[icon_name]
        
        # Determine ROI
        roi = self._get_roi_for_region(selector.region_hint)
        
        # Extract ROI from frame
        roi_img = self._extract_roi(frame.image_bgr, roi)
        if roi_img is None:
            return candidates
        
        # Template matching on edges
        candidates.extend(self._template_match_edges(template, roi_img, roi, frame))
        
        # ORB feature matching as backup
        candidates.extend(self._orb_match(template, roi_img, roi, frame))
        
        return candidates
    
    def _resolve_bbox(self, selector: Target, frame: Frame) -> Candidate:
        """Resolve bbox selector (direct coordinates from LLM)."""
        bbox = selector.bbox
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2
        
        # LLM coordinates are already in active area
        return Candidate((center_x, center_y), selector.confidence, "llm")
    
    def _resolve_point(self, selector: Target, frame: Frame) -> Candidate:
        """Resolve point selector (direct coordinates from LLM)."""
        point_x, point_y = selector.point
        return Candidate((point_x, point_y), selector.confidence, "llm")
    
    def _resolve_region(self, selector: Target, frame: Frame) -> Candidate:
        """Resolve region selector (center of named region)."""
        roi = self._get_roi_for_region(selector.value)
        center_x = roi[0] + roi[2] / 2
        center_y = roi[1] + roi[3] / 2
        
        return Candidate((center_x, center_y), 0.5, "region")
    
    def _get_roi_for_region(self, region_hint: Optional[str]) -> Tuple[float, float, float, float]:
        """Get ROI coordinates for a region hint.
        
        Args:
            region_hint: Named region or None for full image
            
        Returns:
            (x, y, w, h) tuple in normalized coordinates
        """
        if not region_hint:
            return (0.0, 0.0, 1.0, 1.0)
        
        regions = self.config.regions
        
        # Map region hints to actual regions
        region_map = {
            'top_bar': regions.top_bar,
            'bottom_bar': regions.bottom_bar, 
            'left_panel': regions.left_panel,
            'center': regions.center,
            'right_panel': regions.right_panel
        }
        
        return region_map.get(region_hint, (0.0, 0.0, 1.0, 1.0))
    
    def _extract_roi(self, img_bgr: np.ndarray, roi: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """Extract ROI from image."""
        h, w = img_bgr.shape[:2]
        x, y, roi_w, roi_h = roi
        
        # Convert to pixels
        px = int(x * w)
        py = int(y * h)
        pw = int(roi_w * w)
        ph = int(roi_h * h)
        
        # Clamp to bounds
        px = max(0, min(px, w))
        py = max(0, min(py, h))
        pw = min(pw, w - px)
        ph = min(ph, h - py)
        
        if pw <= 0 or ph <= 0:
            return None
        
        return img_bgr[py:py+ph, px:px+pw]
    
    def _template_match_edges(self, template: dict, roi_img: np.ndarray, roi: Tuple[float, float, float, float], frame: Frame) -> List[Candidate]:
        """Perform template matching on edge maps."""
        candidates = []
        
        template_edges = template['edges']
        th, tw = template_edges.shape
        
        # Convert ROI to grayscale and extract edges
        roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        roi_edges = cv2.Canny(roi_gray, 50, 150)
        
        # Multi-scale matching
        scales = np.arange(0.7, 1.31, 0.05)
        
        for scale in scales:
            # Resize template
            new_w = int(tw * scale)
            new_h = int(th * scale)
            
            if new_w <= 0 or new_h <= 0 or new_w >= roi_edges.shape[1] or new_h >= roi_edges.shape[0]:
                continue
            
            scaled_template = cv2.resize(template_edges, (new_w, new_h))
            
            # Template matching
            result = cv2.matchTemplate(roi_edges, scaled_template, cv2.TM_CCOEFF_NORMED)
            
            # Find peaks
            locations = np.where(result >= self.config.thresholds.ncc_edge)
            
            for pt_y, pt_x in zip(locations[0], locations[1]):
                conf = float(result[pt_y, pt_x])
                
                # Calculate center point
                center_x_roi = (pt_x + new_w / 2) / roi_img.shape[1]
                center_y_roi = (pt_y + new_h / 2) / roi_img.shape[0]
                
                # Convert to active area coordinates
                roi_x, roi_y, roi_w, roi_h = roi
                center_x = roi_x + center_x_roi * roi_w
                center_y = roi_y + center_y_roi * roi_h
                
                candidates.append(Candidate((center_x, center_y), conf, "template"))
        
        return candidates
    
    def _orb_match(self, template: dict, roi_img: np.ndarray, roi: Tuple[float, float, float, float], frame: Frame) -> List[Candidate]:
        """Perform ORB feature matching."""
        candidates = []
        
        try:
            # Initialize ORB detector
            orb = cv2.ORB_create(nfeatures=1500)
            
            # Detect features in template
            template_gray = template['gray']
            kp1, des1 = orb.detectAndCompute(template_gray, None)
            
            if des1 is None:
                return candidates
            
            # Detect features in ROI
            roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            kp2, des2 = orb.detectAndCompute(roi_gray, None)
            
            if des2 is None:
                return candidates
            
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
            
            # Need enough matches for homography
            if len(good_matches) >= self.config.thresholds.orb_inliers:
                # Extract matched points
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                
                if M is not None:
                    inliers = np.sum(mask)
                    if inliers >= self.config.thresholds.orb_inliers:
                        # Calculate template center in ROI
                        th, tw = template_gray.shape
                        template_center = np.array([[[tw/2, th/2]]], dtype=np.float32)
                        roi_center = cv2.perspectiveTransform(template_center, M)[0][0]
                        
                        # Convert to normalized coordinates
                        center_x_roi = roi_center[0] / roi_img.shape[1]
                        center_y_roi = roi_center[1] / roi_img.shape[0]
                        
                        # Convert to active area coordinates
                        roi_x, roi_y, roi_w, roi_h = roi
                        center_x = roi_x + center_x_roi * roi_w
                        center_y = roi_y + center_y_roi * roi_h
                        
                        conf = min(1.0, inliers / 50.0)  # Normalize inlier count to confidence
                        candidates.append(Candidate((center_x, center_y), conf, "orb"))
        
        except Exception as e:
            logger.debug(f"ORB matching failed: {e}")
        
        return candidates
    
    def _text_match_score(self, target: str, candidate: str) -> float:
        """Calculate text match score using synonyms and fuzzy matching."""
        target_lower = target.lower()
        candidate_lower = candidate.lower()
        
        # Direct match
        if target_lower == candidate_lower:
            return 1.0
        
        # Check synonyms
        for key, synonyms in self._synonyms.items():
            if target in synonyms:
                for synonym in synonyms:
                    if synonym.lower() == candidate_lower:
                        return 0.95
        
        # Fuzzy string matching
        return fuzz.WRatio(target_lower, candidate_lower) / 100.0
    
    def _is_valid_point(self, point: Tuple[float, float]) -> bool:
        """Check if point is within valid bounds."""
        x, y = point
        
        # Check bounds (with small margin)
        margin = 0.01
        if x < margin or x > (1.0 - margin) or y < margin or y > (1.0 - margin):
            return False
        
        return True
    
    def _convert_full_to_active(self, full_x: float, full_y: float, frame: Frame) -> Tuple[float, float]:
        """Convert full image coordinates to active area coordinates."""
        ax, ay, aw, ah = frame.active_rect
        
        # Convert from full image normalized to pixels
        full_px = full_x * frame.full_w
        full_py = full_y * frame.full_h
        
        # Convert to active area pixels
        active_px = full_px - ax
        active_py = full_py - ay
        
        # Convert to active area normalized
        active_x = active_px / aw if aw > 0 else 0.0
        active_y = active_py / ah if ah > 0 else 0.0
        
        return (active_x, active_y)
    
    def _fuse_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        """Apply confidence fusion for candidates that agree."""
        if len(candidates) <= 1:
            return candidates
        
        fused = []
        used = set()
        
        for i, c1 in enumerate(candidates):
            if i in used:
                continue
            
            agreeing = [c1]
            used.add(i)
            
            for j, c2 in enumerate(candidates[i+1:], i+1):
                if j in used:
                    continue
                
                # Check if candidates agree (within 2% distance)
                dist = ((c1.point[0] - c2.point[0])**2 + (c1.point[1] - c2.point[1])**2)**0.5
                if dist < 0.02:  # 2% of screen
                    agreeing.append(c2)
                    used.add(j)
            
            if len(agreeing) > 1:
                # Average position and boost confidence
                avg_x = sum(c.point[0] for c in agreeing) / len(agreeing)
                avg_y = sum(c.point[1] for c in agreeing) / len(agreeing)
                avg_conf = sum(c.confidence for c in agreeing) / len(agreeing)
                boosted_conf = min(1.0, avg_conf + 0.1)  # 0.1 bonus for agreement
                
                methods = "+".join(set(c.method for c in agreeing))
                fused.append(Candidate((avg_x, avg_y), boosted_conf, methods))
            else:
                fused.append(c1)
        
        return fused
    
    def to_pixels(self, point_norm: Tuple[float, float], frame: Frame) -> Tuple[int, int]:
        """Convert normalized active area coordinates to full frame pixels.
        
        Args:
            point_norm: Normalized coordinates in active area
            frame: Frame containing active area info
            
        Returns:
            Pixel coordinates in full frame
        """
        x_norm, y_norm = point_norm
        ax, ay, aw, ah = frame.active_rect
        
        # Convert to active area pixels
        active_x = int(x_norm * aw)
        active_y = int(y_norm * ah)
        
        # Convert to full frame pixels
        full_x = ax + active_x
        full_y = ay + active_y
        
        return (full_x, full_y)