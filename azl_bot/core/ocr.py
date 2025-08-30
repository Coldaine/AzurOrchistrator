"""OCR client for text extraction from game UI."""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from loguru import logger

from .configs import ResolverConfig


class OCRClient:
    """OCR client supporting multiple backends."""
    
    def __init__(self, config: ResolverConfig) -> None:
        """Initialize OCR client.
        
        Args:
            config: Resolver configuration containing OCR settings
        """
        self.config = config
        self.backend = config.ocr
        self._ocr_engine = None
        self._init_backend()
    
    def _init_backend(self) -> None:
        """Initialize the OCR backend."""
        try:
            if self.backend == "paddle":
                self._init_paddle()
            elif self.backend == "tesseract":
                self._init_tesseract()
            else:
                raise ValueError(f"Unsupported OCR backend: {self.backend}")
                
            logger.info(f"Initialized OCR backend: {self.backend}")
            
        except ImportError as e:
            logger.warning(f"Failed to initialize {self.backend} OCR: {e}")
            if self.backend == "paddle":
                logger.info("Falling back to tesseract")
                self.backend = "tesseract"
                self._init_tesseract()
            else:
                raise
    
    def _init_paddle(self) -> None:
        """Initialize PaddleOCR backend."""
        try:
            from paddleocr import PaddleOCR
            
            # Initialize with English and Chinese models
            self._ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang='en',  # Can be changed to support multiple languages
                show_log=False,
                use_gpu=False  # Set to True if GPU is available
            )
            
        except ImportError:
            raise ImportError("PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")
    
    def _init_tesseract(self) -> None:
        """Initialize Tesseract backend."""
        try:
            import pytesseract
            
            # Test that tesseract is available
            pytesseract.get_tesseract_version()
            self._ocr_engine = pytesseract
            
        except ImportError:
            raise ImportError("pytesseract not available. Install with: pip install pytesseract")
        except Exception:
            raise RuntimeError("Tesseract binary not found. Install with: sudo apt-get install tesseract-ocr")
    
    def text_in_roi(self, img_bgr: np.ndarray, roi_norm: tuple[float, float, float, float], 
                   numeric_only: bool = False) -> List[Dict[str, Any]]:
        """Extract text from region of interest.
        
        Args:
            img_bgr: Input image in BGR format
            roi_norm: Region of interest as (x, y, w, h) in normalized coordinates (0-1)
            numeric_only: If True, only extract numeric characters
            
        Returns:
            List of dictionaries with keys: text, conf, box_norm
        """
        h, w = img_bgr.shape[:2]
        
        # Convert normalized ROI to pixel coordinates
        x_norm, y_norm, w_norm, h_norm = roi_norm
        x = int(x_norm * w)
        y = int(y_norm * h)
        roi_w = int(w_norm * w)
        roi_h = int(h_norm * h)
        
        # Clamp to image bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        roi_w = min(roi_w, w - x)
        roi_h = min(roi_h, h - y)
        
        if roi_w <= 0 or roi_h <= 0:
            logger.warning(f"Invalid ROI: {roi_norm}")
            return []
        
        # Extract ROI
        roi_img = img_bgr[y:y+roi_h, x:x+roi_w]
        
        # Preprocess for OCR
        processed_img = self._preprocess_for_ocr(roi_img, numeric_only)
        
        # Run OCR
        if self.backend == "paddle":
            results = self._ocr_paddle(processed_img, numeric_only)
        else:
            results = self._ocr_tesseract(processed_img, numeric_only)
        
        # Convert coordinates back to normalized format relative to full image
        normalized_results = []
        for result in results:
            # Convert box coordinates from ROI to full image, then normalize
            box_roi = result['box_norm']  # This is relative to ROI
            box_full = [
                (x + box_roi[0] * roi_w) / w,  # x
                (y + box_roi[1] * roi_h) / h,  # y  
                box_roi[2] * roi_w / w,        # w
                box_roi[3] * roi_h / h         # h
            ]
            
            normalized_results.append({
                'text': result['text'],
                'conf': result['conf'],
                'box_norm': box_full
            })
        
        logger.debug(f"OCR found {len(normalized_results)} text items in ROI {roi_norm}")
        return normalized_results
    
    def _preprocess_for_ocr(self, img_bgr: np.ndarray, numeric_only: bool) -> np.ndarray:
        """Preprocess image for better OCR results.
        
        Args:
            img_bgr: Input image
            numeric_only: Whether this is for numeric text only
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding 
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Light dilation to connect broken characters
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.dilate(thresh, kernel, iterations=1)
        
        return processed
    
    def _ocr_paddle(self, img: np.ndarray, numeric_only: bool) -> List[Dict[str, Any]]:
        """Run PaddleOCR on preprocessed image.
        
        Args:
            img: Preprocessed image
            numeric_only: Whether to filter for numeric content
            
        Returns:
            List of OCR results
        """
        try:
            # Run OCR
            results = self._ocr_engine.ocr(img, cls=True)
            
            if not results or not results[0]:
                return []
            
            ocr_results = []
            h, w = img.shape[:2]
            
            for line in results[0]:
                if len(line) < 2:
                    continue
                    
                box, (text, conf) = line
                
                if not text or conf < 0.5:
                    continue
                
                # Filter numeric only if requested
                if numeric_only and not self._is_numeric_text(text):
                    continue
                
                # Convert box to normalized coordinates
                box_np = np.array(box)
                x_min = float(np.min(box_np[:, 0])) / w
                y_min = float(np.min(box_np[:, 1])) / h
                x_max = float(np.max(box_np[:, 0])) / w
                y_max = float(np.max(box_np[:, 1])) / h
                
                box_norm = [x_min, y_min, x_max - x_min, y_max - y_min]
                
                ocr_results.append({
                    'text': text.strip(),
                    'conf': float(conf),
                    'box_norm': box_norm
                })
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
            return []
    
    def _ocr_tesseract(self, img: np.ndarray, numeric_only: bool) -> List[Dict[str, Any]]:
        """Run Tesseract OCR on preprocessed image.
        
        Args:
            img: Preprocessed image
            numeric_only: Whether to filter for numeric content
            
        Returns:
            List of OCR results
        """
        try:
            import pytesseract
            
            # Configure Tesseract
            config = '--psm 6'  # Uniform block of text
            if numeric_only:
                config += ' -c tessedit_char_whitelist=0123456789:'
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
            
            ocr_results = []
            h, w = img.shape[:2]
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = float(data['conf'][i])
                
                if not text or conf < 50:  # Tesseract confidence is 0-100
                    continue
                
                # Filter numeric only if requested
                if numeric_only and not self._is_numeric_text(text):
                    continue
                
                # Get bounding box
                x = float(data['left'][i]) / w
                y = float(data['top'][i]) / h
                box_w = float(data['width'][i]) / w
                box_h = float(data['height'][i]) / h
                
                box_norm = [x, y, box_w, box_h]
                
                ocr_results.append({
                    'text': text,
                    'conf': conf / 100.0,  # Convert to 0-1 range
                    'box_norm': box_norm
                })
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return []
    
    def _is_numeric_text(self, text: str) -> bool:
        """Check if text contains primarily numeric content.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is primarily numeric
        """
        # Allow digits, colons (for time), commas, periods, and whitespace
        allowed_chars = set('0123456789:,. ')
        text_chars = set(text)
        
        # At least 70% of characters should be numeric-related
        if not text_chars:
            return False
            
        numeric_chars = text_chars.intersection(set('0123456789'))
        return len(numeric_chars) / len(text_chars) >= 0.3