"""Daily maintenance task - orchestrates multiple daily activities."""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger

from ..core.capture import Frame


class StepStatus(str, Enum):
    """Status of a step execution."""
    SUCCESS = "success"
    ALREADY_COMPLETE = "already_complete"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result of executing a step."""
    step_name: str
    status: StepStatus
    confidence: float
    details: Optional[str] = None
    duration: float = 0.0


class DailyMaintenanceTask:
    """Orchestrates daily maintenance activities in a safe, idempotent manner.
    
    This task chains multiple sub-tasks:
    - go_home: Navigate to home screen
    - collect_mailbox: Collect mail/pickups
    - collect_commissions: Check and collect commissions
    - record_currencies: Record current currency balances
    
    The task is designed to be idempotent - safe to re-run even if partially
    complete. Each step gracefully handles "already complete" states.
    """
    
    name = "daily_maintenance"
    
    def __init__(self):
        """Initialize daily maintenance task."""
        self.results: List[StepResult] = []
    
    def goal(self) -> Dict[str, Any]:
        """Return goal description for LLM.
        
        Returns:
            Goal description dict
        """
        return {
            "action": "daily_maintenance",
            "description": (
                "Perform daily maintenance: Navigate to Home, collect all mail/missions, "
                "check commissions, and record currency balances."
            )
        }
    
    def success(self, frame: Frame, context: Dict[str, Any]) -> bool:
        """Check if daily maintenance was successful.
        
        Args:
            frame: Current frame
            context: Context including OCR results
            
        Returns:
            True if all steps completed successfully or were already complete
        """
        if not self.results:
            return False
        
        # Success if all steps are either SUCCESS or ALREADY_COMPLETE
        return all(
            r.status in (StepStatus.SUCCESS, StepStatus.ALREADY_COMPLETE)
            for r in self.results
        )
    
    def on_success(self, planner, frame: Frame) -> None:
        """Handle successful completion.
        
        Args:
            planner: Planner instance
            frame: Current frame
        """
        logger.info("✓ Daily maintenance completed successfully!")
        
        # Log summary
        for result in self.results:
            status_icon = "✓" if result.status == StepStatus.SUCCESS else "↻"
            logger.info(
                f"  {status_icon} {result.step_name}: {result.status.value} "
                f"(confidence: {result.confidence:.2f}, duration: {result.duration:.1f}s)"
            )
    
    def execute(self, planner) -> List[StepResult]:
        """Execute the daily maintenance sequence.
        
        This is the main entry point that orchestrates all steps.
        
        Args:
            planner: Planner instance with access to all components
            
        Returns:
            List of step results
        """
        self.results = []
        
        logger.info("Starting daily maintenance...")
        
        # Step 1: Go home
        self.results.append(self._go_home(planner))
        
        # Step 2: Collect mailbox/pickups
        self.results.append(self._collect_mailbox(planner))
        
        # Step 3: Check commissions
        self.results.append(self._check_commissions(planner))
        
        # Step 4: Record currencies
        self.results.append(self._record_currencies(planner))
        
        return self.results
    
    def _go_home(self, planner) -> StepResult:
        """Navigate to home screen.
        
        Args:
            planner: Planner instance
            
        Returns:
            Step result
        """
        step_name = "go_home"
        start_time = time.time()
        
        try:
            logger.info(f"Step: {step_name}")
            
            # Capture current frame
            frame = planner.capture.grab()
            
            # Run OCR to identify current screen
            ocr_results = planner.ocr.text_in_roi(frame.image_bgr, (0, 0, 1, 1))
            
            # Check if already on home screen
            from ..core.screens import identify_screen
            current_screen = identify_screen(ocr_results)
            
            if current_screen == "home":
                logger.info("Already on home screen")
                duration = time.time() - start_time
                return StepResult(
                    step_name=step_name,
                    status=StepStatus.ALREADY_COMPLETE,
                    confidence=0.9,
                    details="Already on home screen",
                    duration=duration
                )
            
            # Navigate to home - press back multiple times if needed
            max_attempts = 3
            for attempt in range(max_attempts):
                planner.device.key_back()
                time.sleep(1.5)
                
                # Check if we reached home
                frame = planner.capture.grab()
                ocr_results = planner.ocr.text_in_roi(frame.image_bgr, (0, 0, 1, 1))
                current_screen = identify_screen(ocr_results)
                
                if current_screen == "home":
                    duration = time.time() - start_time
                    return StepResult(
                        step_name=step_name,
                        status=StepStatus.SUCCESS,
                        confidence=0.85,
                        details=f"Reached home screen after {attempt + 1} back presses",
                        duration=duration
                    )
            
            # Failed to reach home
            duration = time.time() - start_time
            return StepResult(
                step_name=step_name,
                status=StepStatus.FAILED,
                confidence=0.3,
                details="Could not reach home screen",
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error in {step_name}: {e}")
            return StepResult(
                step_name=step_name,
                status=StepStatus.FAILED,
                confidence=0.0,
                details=f"Exception: {str(e)}",
                duration=duration
            )
    
    def _collect_mailbox(self, planner) -> StepResult:
        """Collect mail and pickups.
        
        Args:
            planner: Planner instance
            
        Returns:
            Step result
        """
        step_name = "collect_mailbox"
        start_time = time.time()
        
        try:
            logger.info(f"Step: {step_name}")
            
            # Use the pickups task
            from .pickups import create_pickups_task
            pickups_task = create_pickups_task()
            
            # Capture frame and check for pickups
            frame = planner.capture.grab()
            ocr_results = planner.ocr.text_in_roi(frame.image_bgr, (0, 0, 1, 1))
            
            # Detect red badges
            from ..core.screens import detect_red_badges
            badges = detect_red_badges(frame)
            
            if not badges:
                logger.info("No pickups detected")
                duration = time.time() - start_time
                return StepResult(
                    step_name=step_name,
                    status=StepStatus.ALREADY_COMPLETE,
                    confidence=0.8,
                    details="No pickups available",
                    duration=duration
                )
            
            # Run the pickups task
            success = planner.run_task(pickups_task)
            duration = time.time() - start_time
            
            if success:
                return StepResult(
                    step_name=step_name,
                    status=StepStatus.SUCCESS,
                    confidence=0.85,
                    details=f"Collected {len(badges)} pickups",
                    duration=duration
                )
            else:
                return StepResult(
                    step_name=step_name,
                    status=StepStatus.FAILED,
                    confidence=0.4,
                    details="Pickup collection failed",
                    duration=duration
                )
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error in {step_name}: {e}")
            return StepResult(
                step_name=step_name,
                status=StepStatus.FAILED,
                confidence=0.0,
                details=f"Exception: {str(e)}",
                duration=duration
            )
    
    def _check_commissions(self, planner) -> StepResult:
        """Check commission status and collect if ready.
        
        Args:
            planner: Planner instance
            
        Returns:
            Step result
        """
        step_name = "check_commissions"
        start_time = time.time()
        
        try:
            logger.info(f"Step: {step_name}")
            
            # Use the commissions task
            from .commissions import create_commissions_task
            commissions_task = create_commissions_task()
            
            # Run the commissions reading task
            success = planner.run_task(commissions_task)
            duration = time.time() - start_time
            
            if success:
                return StepResult(
                    step_name=step_name,
                    status=StepStatus.SUCCESS,
                    confidence=0.85,
                    details="Commission status checked",
                    duration=duration
                )
            else:
                return StepResult(
                    step_name=step_name,
                    status=StepStatus.FAILED,
                    confidence=0.4,
                    details="Could not check commissions",
                    duration=duration
                )
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error in {step_name}: {e}")
            return StepResult(
                step_name=step_name,
                status=StepStatus.FAILED,
                confidence=0.0,
                details=f"Exception: {str(e)}",
                duration=duration
            )
    
    def _record_currencies(self, planner) -> StepResult:
        """Record current currency balances.
        
        Args:
            planner: Planner instance
            
        Returns:
            Step result
        """
        step_name = "record_currencies"
        start_time = time.time()
        
        try:
            logger.info(f"Step: {step_name}")
            
            # Use the currencies task
            from .currencies import create_currencies_task
            currencies_task = create_currencies_task()
            
            # Run the currencies reading task
            success = planner.run_task(currencies_task)
            duration = time.time() - start_time
            
            if success:
                return StepResult(
                    step_name=step_name,
                    status=StepStatus.SUCCESS,
                    confidence=0.85,
                    details="Currencies recorded",
                    duration=duration
                )
            else:
                return StepResult(
                    step_name=step_name,
                    status=StepStatus.FAILED,
                    confidence=0.4,
                    details="Could not record currencies",
                    duration=duration
                )
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error in {step_name}: {e}")
            return StepResult(
                step_name=step_name,
                status=StepStatus.FAILED,
                confidence=0.0,
                details=f"Exception: {str(e)}",
                duration=duration
            )


def create_daily_task() -> DailyMaintenanceTask:
    """Create and return a daily maintenance task instance."""
    return DailyMaintenanceTask()
