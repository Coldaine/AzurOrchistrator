"""
Actuator Extensions for Action Verification

This file documents the intended extensions to azl_bot/core/actuator.py.
Due to repository file corruption, the complete implementation cannot be integrated.

INTENDED CHANGES TO Actuator CLASS:
=====================================

1. Add postcondition support to actions:

class Actuator:
    # ... existing code ...
    
    def __init__(self, device: Device, backend: Literal["adb", "minitouch"] = "adb") -> None:
        # ... existing initialization ...
        self.capture = None  # Will be set by bootstrap
        self._verification_enabled = True
        
    def tap_norm_verified(
        self, 
        x: float, 
        y: float, 
        active_rect: tuple[int, int, int, int] | None = None,
        postcondition: Optional[Callable[[Frame], bool]] = None,
        expected_selector: Optional[Target] = None
    ) -> bool:
        '''
        Tap with verification support.
        
        Args:
            x: Normalized x coordinate
            y: Normalized y coordinate
            active_rect: Active area for coordinate transformation
            postcondition: Optional callback to verify action success
            expected_selector: Optional target selector to verify appeared/disappeared
            
        Returns:
            True if verification passed (or verification disabled), False otherwise
        '''
        # Perform tap
        self.tap_norm(x, y, active_rect)
        
        if not self._verification_enabled:
            return True
        
        # Wait for frame to stabilize
        if postcondition and self.capture:
            time.sleep(0.5)  # Brief wait for UI to update
            frame = self.capture.grab()
            try:
                return postcondition(frame)
            except Exception as e:
                logger.error(f"Postcondition verification failed: {e}")
                return False
        
        return True  # No verification requested
    
    def swipe_norm_verified(
        self,
        x1: float, 
        y1: float, 
        x2: float, 
        y2: float,
        ms: int = 200,
        active_rect: tuple[int, int, int, int] | None = None,
        postcondition: Optional[Callable[[Frame], bool]] = None
    ) -> bool:
        '''
        Swipe with verification support.
        
        Args:
            x1, y1: Start normalized coordinates
            x2, y2: End normalized coordinates
            ms: Swipe duration
            active_rect: Active area for coordinate transformation
            postcondition: Optional callback to verify action success
            
        Returns:
            True if verification passed, False otherwise
        '''
        # Perform swipe
        self.swipe_norm(x1, y1, x2, y2, ms, active_rect)
        
        if not self._verification_enabled:
            return True
        
        # Wait for frame to stabilize
        if postcondition and self.capture:
            time.sleep(max(0.5, ms / 1000.0 + 0.3))
            frame = self.capture.grab()
            try:
                return postcondition(frame)
            except Exception as e:
                logger.error(f"Postcondition verification failed: {e}")
                return False
        
        return True
    
    def verify(
        self, 
        postcondition: Callable[[Frame], bool],
        timeout_sec: float = 5.0
    ) -> bool:
        '''
        Standalone verification hook for external use.
        
        Args:
            postcondition: Callback that returns True if verification succeeds
            timeout_sec: Maximum time to wait for condition
            
        Returns:
            True if condition met within timeout, False otherwise
        '''
        if not self.capture:
            logger.warning("No capture available for verification")
            return False
        
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            frame = self.capture.grab()
            try:
                if postcondition(frame):
                    return True
            except Exception as e:
                logger.debug(f"Verification check raised: {e}")
            
            time.sleep(0.2)  # Check frequency
        
        logger.warning(f"Verification timeout after {timeout_sec}s")
        return False


USAGE EXAMPLE:
==============

# In planner or loop code:
from azl_bot.core.llm_client import Target

# Define postcondition: "OK button should disappear"
def ok_button_gone(frame):
    from azl_bot.core.resolver import Resolver
    resolver = get_resolver()  # From context
    target = Target(kind="text", value="OK", region_hint="center")
    result = resolver.resolve(target, frame)
    return result is None or result.confidence < 0.5

# Execute tap with verification
success = actuator.tap_norm_verified(
    0.5, 0.8,
    postcondition=ok_button_gone
)

if not success:
    logger.warning("OK button did not disappear as expected")
    # Trigger retry or recovery

INTEGRATION WITH StateLoop:
============================

The StateLoop.execute_with_retry() method already supports postconditions.
The Actuator extensions provide a lower-level way to add verification 
directly to individual taps/swipes when not using the loop.

# Example: Using StateLoop with Actuator
def tap_action():
    actuator.tap_norm(0.5, 0.5, frame.active_rect)

success, frame = loop.execute_with_retry(
    tap_action,
    postcondition=lambda f: verify_ui_changed(f)
)
"""
