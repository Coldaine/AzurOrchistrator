"""State loop implementation for deterministic Sense → Think → Act → Check cycle."""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol

import numpy as np
from loguru import logger

from .hashing import FrameHasher


@dataclass
class LoopConfig:
    """Configuration for StateLoop."""
    target_fps: float = 2.0
    stability_frames: int = 3
    stability_timeout_sec: float = 10.0
    max_retries: int = 3
    retry_backoff_base: float = 1.5
    recovery_enabled: bool = True
    hamming_threshold: float = 0.05  # As fraction of hash_size^2


@dataclass
class LoopMetrics:
    """Telemetry metrics for loop execution."""
    actions_attempted: int = 0
    actions_succeeded: int = 0
    actions_retried: int = 0
    recoveries_triggered: int = 0
    failures: int = 0
    total_resolve_time_sec: float = 0.0
    resolve_count: int = 0
    
    @property
    def avg_resolve_time(self) -> float:
        """Average resolution time in seconds."""
        if self.resolve_count == 0:
            return 0.0
        return self.total_resolve_time_sec / self.resolve_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "actions_attempted": self.actions_attempted,
            "actions_succeeded": self.actions_succeeded,
            "actions_retried": self.actions_retried,
            "recoveries_triggered": self.recoveries_triggered,
            "failures": self.failures,
            "avg_resolve_time_sec": self.avg_resolve_time,
            "resolve_count": self.resolve_count,
        }


class Frame(Protocol):
    """Protocol for frame objects."""
    image_bgr: np.ndarray
    active_rect: Optional[tuple[int, int, int, int]]
    full_w: int
    full_h: int


class CaptureProtocol(Protocol):
    """Protocol for capture component."""
    def grab(self) -> Frame: ...


class ActuatorProtocol(Protocol):
    """Protocol for actuator component."""
    def tap_norm(self, x: float, y: float, active_rect: Optional[tuple[int, int, int, int]] = None) -> None: ...
    def swipe_norm(self, x1: float, y1: float, x2: float, y2: float, ms: int = 200, active_rect: Optional[tuple[int, int, int, int]] = None) -> None: ...


class DeviceProtocol(Protocol):
    """Protocol for device component."""
    def key_back(self) -> None: ...
    def key_home(self) -> None: ...


class StateLoop:
    """Deterministic Sense → Think → Act → Check loop with verification and recovery."""
    
    def __init__(
        self,
        config: LoopConfig,
        capture: CaptureProtocol,
        actuator: ActuatorProtocol,
        device: DeviceProtocol,
        hasher: Optional[FrameHasher] = None,
    ):
        """Initialize StateLoop.
        
        Args:
            config: Loop configuration
            capture: Capture component for grabbing frames
            actuator: Actuator component for performing actions
            device: Device component for system controls
            hasher: Optional FrameHasher for stability checks (creates new if None)
        """
        self.config = config
        self.capture = capture
        self.actuator = actuator
        self.device = device
        self.hasher = hasher or FrameHasher(similarity_threshold=0.95)
        self.metrics = LoopMetrics()
        
        self._last_frame_time = 0.0
        self._running = False
    
    def wait_for_stability(
        self,
        timeout_sec: Optional[float] = None,
        required_frames: Optional[int] = None
    ) -> tuple[bool, Optional[Frame]]:
        """Wait for frame to become stable (pre/post action).
        
        Args:
            timeout_sec: Maximum time to wait (uses config default if None)
            required_frames: Number of stable frames required (uses config default if None)
            
        Returns:
            Tuple of (is_stable, last_frame). is_stable is True if stability achieved,
            False if timeout. last_frame is the final frame captured (or None on error).
        """
        timeout_sec = timeout_sec if timeout_sec is not None else self.config.stability_timeout_sec
        required_frames = required_frames if required_frames is not None else self.config.stability_frames
        
        start_time = time.time()
        last_frame = None
        
        logger.debug(f"Waiting for stability: {required_frames} frames, timeout={timeout_sec}s")
        
        while time.time() - start_time < timeout_sec:
            try:
                frame = self.capture.grab()
                last_frame = frame
                
                if self.hasher.is_stable(frame.image_bgr, required_matches=required_frames):
                    elapsed = time.time() - start_time
                    logger.info(f"Frame stable after {elapsed:.2f}s ({required_frames} frames)")
                    return True, frame
                
                # Respect target FPS
                self._maintain_fps()
                
            except Exception as e:
                logger.error(f"Error during stability check: {e}")
                return False, last_frame
        
        logger.warning(f"Stability timeout after {timeout_sec}s")
        return False, last_frame
    
    def verify_action(
        self,
        postcondition: Optional[Callable[[Frame], bool]] = None,
        expected_change: bool = True
    ) -> tuple[bool, Optional[Frame]]:
        """Verify action postcondition.
        
        Args:
            postcondition: Optional callback that returns True if action succeeded
            expected_change: If True, expects frame to change; if False, expects no change
            
        Returns:
            Tuple of (success, frame). success is True if verification passed.
        """
        # Wait a moment for action to take effect
        time.sleep(0.3)
        
        # Wait for post-action stability
        stable, frame = self.wait_for_stability()
        
        if not stable:
            logger.warning("Post-action frame did not stabilize")
            return False, frame
        
        # If postcondition provided, check it
        if postcondition and frame:
            try:
                result = postcondition(frame)
                if not result:
                    logger.warning("Postcondition check failed")
                return result, frame
            except Exception as e:
                logger.error(f"Postcondition check raised exception: {e}")
                return False, frame
        
        # Default: assume success if frame stabilized
        return True, frame
    
    def execute_with_retry(
        self,
        action: Callable[[], None],
        postcondition: Optional[Callable[[Frame], bool]] = None,
        max_retries: Optional[int] = None
    ) -> tuple[bool, Optional[Frame]]:
        """Execute action with bounded retries and exponential backoff.
        
        Args:
            action: Callable that performs the action
            postcondition: Optional verification callback
            max_retries: Maximum retry attempts (uses config default if None)
            
        Returns:
            Tuple of (success, final_frame)
        """
        max_retries = max_retries if max_retries is not None else self.config.max_retries
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            self.metrics.actions_attempted += 1
            
            if attempt > 0:
                self.metrics.actions_retried += 1
                backoff = self.config.retry_backoff_base ** attempt
                logger.info(f"Retry attempt {attempt}/{max_retries} after {backoff:.2f}s backoff")
                time.sleep(backoff)
            
            try:
                # Wait for pre-action stability
                stable, _ = self.wait_for_stability()
                if not stable:
                    logger.warning(f"Pre-action stability timeout (attempt {attempt + 1})")
                    if attempt == max_retries:
                        break
                    continue
                
                # Execute the action
                logger.debug(f"Executing action (attempt {attempt + 1})")
                action()
                
                # Verify postcondition
                success, frame = self.verify_action(postcondition=postcondition)
                
                if success:
                    self.metrics.actions_succeeded += 1
                    logger.info(f"Action succeeded (attempt {attempt + 1})")
                    return True, frame
                else:
                    logger.warning(f"Action verification failed (attempt {attempt + 1})")
                    
            except Exception as e:
                logger.error(f"Action execution failed (attempt {attempt + 1}): {e}")
        
        # All attempts exhausted
        self.metrics.failures += 1
        logger.error(f"Action failed after {max_retries + 1} attempts")
        return False, None
    
    def recovery(self, recovery_sequence: Optional[list[str]] = None) -> bool:
        """Execute recovery routine.
        
        Args:
            recovery_sequence: List of recovery actions ("back", "home", etc.)
                             Uses default if None: ["back", "back", "back", "home"]
                             
        Returns:
            True if recovery sequence completed (does not guarantee success)
        """
        if not self.config.recovery_enabled:
            logger.info("Recovery disabled in config")
            return False
        
        self.metrics.recoveries_triggered += 1
        logger.warning("Initiating recovery sequence")
        
        recovery_sequence = recovery_sequence or ["back", "back", "back", "home"]
        
        for action in recovery_sequence:
            try:
                logger.info(f"Recovery action: {action}")
                if action == "back":
                    self.device.key_back()
                elif action == "home":
                    self.device.key_home()
                else:
                    logger.warning(f"Unknown recovery action: {action}")
                
                time.sleep(1.0)  # Wait between recovery actions
                
            except Exception as e:
                logger.error(f"Recovery action '{action}' failed: {e}")
        
        logger.info("Recovery sequence completed")
        return True
    
    def run_action_with_recovery(
        self,
        action: Callable[[], None],
        postcondition: Optional[Callable[[Frame], bool]] = None,
        recovery_sequence: Optional[list[str]] = None
    ) -> tuple[bool, Optional[Frame]]:
        """Execute action with retry and recovery on failure.
        
        Args:
            action: Callable that performs the action
            postcondition: Optional verification callback
            recovery_sequence: Optional custom recovery sequence
            
        Returns:
            Tuple of (success, final_frame)
        """
        # Try with retries
        success, frame = self.execute_with_retry(action, postcondition)
        
        # If failed, attempt recovery
        if not success and self.config.recovery_enabled:
            logger.warning("Action failed, attempting recovery")
            self.recovery(recovery_sequence)
            
            # Optionally retry once after recovery
            logger.info("Retrying action after recovery")
            success, frame = self.execute_with_retry(action, postcondition, max_retries=1)
        
        return success, frame
    
    def _maintain_fps(self):
        """Maintain target FPS by sleeping if needed."""
        current_time = time.time()
        target_interval = 1.0 / self.config.target_fps
        
        if self._last_frame_time > 0:
            elapsed = current_time - self._last_frame_time
            if elapsed < target_interval:
                sleep_time = target_interval - elapsed
                time.sleep(sleep_time)
        
        self._last_frame_time = time.time()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current loop metrics.
        
        Returns:
            Dictionary of metric values
        """
        return self.metrics.to_dict()
    
    def reset_metrics(self):
        """Reset all metrics counters."""
        self.metrics = LoopMetrics()
        logger.debug("Loop metrics reset")
