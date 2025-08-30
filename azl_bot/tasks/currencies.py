"""Currency reading task."""

import re
from typing import Any, Dict

from loguru import logger

from ..core.capture import Frame
from ..core.screens import Regions


class CurrenciesTask:
    """Task to read currency balances from the top bar."""
    
    name = "currencies"
    
    def goal(self) -> Dict[str, Any]:
        """Return goal description for LLM."""
        return {
            "action": "read_currencies",
            "description": "Read Oil, Coins, and Gems from the top bar. Navigate to Home if not already there."
        }
    
    def success(self, frame: Frame, context: Dict[str, Any]) -> bool:
        """Check if currency reading was successful.
        
        Args:
            frame: Current frame
            context: Context including OCR results
            
        Returns:
            True if currencies were successfully extracted
        """
        # Extract currencies from top bar
        currencies = self._extract_currencies(frame, context.get("ocr_results", []))
        
        # Success if we got at least 2 of the 3 main currencies
        extracted_count = sum(1 for value in [currencies["oil"], currencies["coins"], currencies["gems"]] if value is not None)
        
        return extracted_count >= 2
    
    def on_success(self, planner, frame: Frame) -> None:
        """Handle successful currency extraction.
        
        Args:
            planner: Planner instance for database access
            frame: Current frame
        """
        # Run OCR again to get fresh results
        ocr_results = planner.ocr.text_in_roi(frame.image_bgr, (0, 0, 1, 1))
        
        # Extract currencies
        currencies = self._extract_currencies(frame, ocr_results)
        
        # Store in database
        planner.datastore.record_currencies(
            oil=currencies["oil"],
            coins=currencies["coins"], 
            gems=currencies["gems"],
            cubes=currencies["cubes"]
        )
        
        logger.info(f"Recorded currencies: {currencies}")
    
    def _extract_currencies(self, frame: Frame, ocr_results: list) -> Dict[str, int]:
        """Extract currency values from OCR results.
        
        Args:
            frame: Current frame
            ocr_results: OCR results from full frame
            
        Returns:
            Dictionary with currency values (None if not found)
        """
        currencies = {
            "oil": None,
            "coins": None,
            "gems": None,
            "cubes": None
        }
        
        # Filter OCR results to top bar region
        top_bar_results = self._filter_ocr_to_region(ocr_results, Regions.top_bar)
        
        # Look for numeric values near currency indicators
        for result in top_bar_results:
            text = result["text"].strip()
            
            # Clean and extract numbers
            numbers = self._extract_numbers(text)
            if not numbers:
                continue
            
            # Try to identify which currency this represents
            # This is a simplified approach - in practice you'd use more sophisticated matching
            text_lower = text.lower()
            
            if any(indicator in text_lower for indicator in ["oil", "fuel", "燃料"]):
                currencies["oil"] = numbers[0]
            elif any(indicator in text_lower for indicator in ["coin", "gold", "金币", "資金"]):
                currencies["coins"] = numbers[0]  
            elif any(indicator in text_lower for indicator in ["gem", "diamond", "钻石", "ダイヤ"]):
                currencies["gems"] = numbers[0]
            elif any(indicator in text_lower for indicator in ["cube", "立方", "キューブ"]):
                currencies["cubes"] = numbers[0]
            else:
                # If no clear indicator, try positional matching
                # This is where template matching for currency icons would be helpful
                pass
        
        # Alternative approach: look for large numbers in specific regions of top bar
        # This handles cases where OCR doesn't capture currency labels
        if sum(1 for v in currencies.values() if v is not None) < 2:
            currencies.update(self._extract_by_position(top_bar_results))
        
        return currencies
    
    def _filter_ocr_to_region(self, ocr_results: list, region: tuple) -> list:
        """Filter OCR results to those within a specific region.
        
        Args:
            ocr_results: Full OCR results
            region: (x, y, w, h) normalized region
            
        Returns:
            Filtered OCR results
        """
        rx, ry, rw, rh = region
        filtered = []
        
        for result in ocr_results:
            box = result["box_norm"]
            box_center_x = box[0] + box[2] / 2
            box_center_y = box[1] + box[3] / 2
            
            # Check if center point is within region
            if (rx <= box_center_x <= (rx + rw) and 
                ry <= box_center_y <= (ry + rh)):
                filtered.append(result)
        
        return filtered
    
    def _extract_numbers(self, text: str) -> list:
        """Extract numeric values from text.
        
        Args:
            text: Text to extract numbers from
            
        Returns:
            List of extracted integers
        """
        numbers = []
        
        # Remove common formatting and keep only digits and separators
        cleaned = re.sub(r'[^\d,\.\s]', '', text)
        
        # Find number patterns
        # Handle common formats: 1234, 1,234, 1.234, etc.
        number_patterns = [
            r'\b\d{1,3}(?:,\d{3})*\b',  # 1,234,567 format
            r'\b\d+\b'  # Simple digits
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, cleaned)
            for match in matches:
                try:
                    # Remove separators and convert to int
                    value = int(match.replace(',', '').replace('.', ''))
                    # Filter out unreasonably small values (likely noise)
                    if value >= 10:
                        numbers.append(value)
                except ValueError:
                    continue
        
        return numbers
    
    def _extract_by_position(self, top_bar_results: list) -> Dict[str, int]:
        """Extract currencies by expected positions in top bar.
        
        Args:
            top_bar_results: OCR results from top bar
            
        Returns:
            Dictionary with positionally-matched currencies
        """
        currencies = {
            "oil": None,
            "coins": None,
            "gems": None,
            "cubes": None
        }
        
        # Find all numeric texts with their positions
        numeric_texts = []
        for result in top_bar_results:
            numbers = self._extract_numbers(result["text"])
            if numbers:
                box = result["box_norm"]
                center_x = box[0] + box[2] / 2
                
                numeric_texts.append({
                    "value": numbers[0],  # Take the first/largest number
                    "x": center_x,
                    "text": result["text"]
                })
        
        # Sort by X position (left to right)
        numeric_texts.sort(key=lambda x: x["x"])
        
        # Typical layout: Oil (left), Coins (center), Gems (right)
        if len(numeric_texts) >= 3:
            currencies["oil"] = numeric_texts[0]["value"]
            currencies["coins"] = numeric_texts[1]["value"] 
            currencies["gems"] = numeric_texts[2]["value"]
        elif len(numeric_texts) == 2:
            # Assume Oil and Coins (most common)
            currencies["oil"] = numeric_texts[0]["value"]
            currencies["coins"] = numeric_texts[1]["value"]
        elif len(numeric_texts) == 1:
            # Single value - could be any currency, default to coins
            currencies["coins"] = numeric_texts[0]["value"]
        
        return currencies