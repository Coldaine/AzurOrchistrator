"""Main planner for task execution and coordination."""

import time
from typing import Any, Dict, List, Optional, Protocol

from loguru import logger

from .actuator import Actuator
from .capture import Capture, Frame
from .datastore import DataStore
from .device import Device
from .llm_client import LLMClient, Plan, Step
from .loggingx import RunLogger
from .ocr import OCRClient
from .resolver import Resolver
from .screens import get_region_info, identify_screen


class Task(Protocol):
    """Protocol for task implementations."""
    name: str
    
    def goal(self) -> Dict[str, Any]:
        """Return goal description for LLM."""
        ...
    
    def success(self, frame: Frame, context: Dict[str, Any]) -> bool:
        """Check if task goal has been achieved."""
        ...
    
    def on_success(self, planner: "Planner", frame: Frame) -> None:
        """Handle successful task completion."""
        ...


class Planner:
    """Main task planner and execution coordinator."""
    
    def __init__(self, device: Device, capture: Capture, resolver: Resolver,
                 ocr: OCRClient, llm: LLMClient, datastore: DataStore, 
                 actuator: Actuator) -> None:
        """Initialize planner with all required components.
        
        Args:
            device: Device interface
            capture: Screen capture interface
            resolver: Selector resolver
            ocr: OCR client
            llm: LLM client
            datastore: Database interface
            actuator: Input actuator
        """
        self.device = device
        self.capture = capture
        self.resolver = resolver
        self.ocr = ocr
        self.llm = llm
        self.datastore = datastore
        self.actuator = actuator
        
        self.run_logger: Optional[RunLogger] = None
        self.current_run_id: Optional[int] = None
        self.last_screen = "unknown"
        self.step_delay = 1.0  # Delay between steps
        
    def run_task(self, task: Task) -> bool:
        """Execute a complete task.
        
        Args:
            task: Task to execute
            
        Returns:
            True if task completed successfully
        """
        logger.info(f"Starting task: {task.name}")
        
        # Create run record
        self.current_run_id = self.datastore.insert_run(task.name, self.device.serial)
        
        try:
            max_attempts = 10
            attempt = 0
            
            while attempt < max_attempts:
                attempt += 1
                logger.info(f"Task attempt {attempt}/{max_attempts}")
                
                # Capture current frame
                frame = self.capture.grab()
                
                # Run OCR to get visible text
                ocr_results = self.ocr.text_in_roi(frame.image_bgr, (0, 0, 1, 1))
                
                # Identify current screen
                current_screen = identify_screen(frame, ocr_results)
                self.last_screen = current_screen
                
                logger.info(f"Current screen: {current_screen}")
                
                # Check if task is already complete
                context = {
                    "device_w": frame.full_w,
                    "device_h": frame.full_h,
                    "regions": get_region_info(),
                    "last_screen": current_screen,
                    "ocr_results": ocr_results
                }
                
                if task.success(frame, context):
                    logger.info(f"Task {task.name} already completed!")
                    task.on_success(self, frame)
                    return True
                
                # Get plan from LLM
                goal = task.goal()
                plan = self.llm.propose_plan(frame, goal, context)
                
                logger.info(f"LLM plan: {len(plan.steps)} steps, done={plan.done}")
                
                # If LLM says we're done, check success condition
                if plan.done:
                    if task.success(frame, context):
                        logger.info(f"Task {task.name} completed successfully!")
                        task.on_success(self, frame)
                        return True
                    else:
                        logger.warning("LLM said done but success condition not met")
                        plan.done = False
                
                # Execute plan steps
                success = True
                for i, step in enumerate(plan.steps):
                    logger.info(f"Executing step {i+1}/{len(plan.steps)}: {step.action}")
                    
                    if not self.run_step(step, frame):
                        logger.warning(f"Step {i+1} failed: {step.action}")
                        success = False
                        break
                    
                    # Brief delay between steps
                    time.sleep(0.5)
                
                # If we successfully executed all steps, check if task is complete
                if success:
                    # Capture again to check result
                    time.sleep(self.step_delay)
                    frame = self.capture.grab()
                    ocr_results = self.ocr.text_in_roi(frame.image_bgr, (0, 0, 1, 1))
                    
                    context.update({
                        "ocr_results": ocr_results,
                        "last_screen": identify_screen(frame, ocr_results)
                    })
                    
                    if task.success(frame, context):
                        logger.info(f"Task {task.name} completed successfully!")
                        task.on_success(self, frame)
                        return True
                
                # Delay before next attempt
                time.sleep(self.step_delay)
            
            logger.error(f"Task {task.name} failed after {max_attempts} attempts")
            return False
            
        except Exception as e:
            logger.error(f"Task {task.name} failed with exception: {e}")
            return False
        
        finally:
            self.current_run_id = None
    
    def run_step(self, step: Step, frame: Frame) -> bool:
        """Execute a single step.
        
        Args:
            step: Step to execute
            frame: Current frame
            
        Returns:
            True if step executed successfully
        """
        start_time = time.time()
        
        try:
            if step.action == "tap":
                return self._execute_tap(step, frame)
            elif step.action == "swipe":
                return self._execute_swipe(step, frame)
            elif step.action == "wait":
                return self._execute_wait(step, frame)
            elif step.action == "back":
                return self._execute_back(step, frame)
            elif step.action == "assert":
                return self._execute_assert(step, frame)
            else:
                logger.error(f"Unknown action: {step.action}")
                return False
                
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return False
        
        finally:
            # Log action
            duration = time.time() - start_time
            logger.debug(f"Step '{step.action}' took {duration:.2f}s")
    
    def _execute_tap(self, step: Step, frame: Frame) -> bool:
        """Execute tap action."""
        if not step.target:
            logger.error("Tap step missing target")
            return False
        
        # Resolve target to coordinates
        candidate = self.resolver.resolve(step.target, frame)
        if not candidate:
            logger.error(f"Could not resolve target: {step.target}")
            return False
        
        logger.info(f"Tapping at {candidate.point} (confidence: {candidate.confidence:.3f})")
        
        # Log action to database
        if self.current_run_id:
            self.datastore.append_action(
                run_id=self.current_run_id,
                screen=self.last_screen,
                action="tap",
                selector_json=step.target.model_dump_json(),
                method=candidate.method,
                point_norm_x=candidate.point[0],
                point_norm_y=candidate.point[1],
                confidence=candidate.confidence,
                success=None  # Will be updated after verification
            )
        
        # Execute tap
        self.actuator.tap_norm(candidate.point[0], candidate.point[1])
        
        # Wait and verify
        time.sleep(self.step_delay)
        return self._verify_step_success(step, frame)
    
    def _execute_swipe(self, step: Step, frame: Frame) -> bool:
        """Execute swipe action."""
        # For simplicity, not fully implemented in this version
        logger.warning("Swipe action not fully implemented")
        return True
    
    def _execute_wait(self, step: Step, frame: Frame) -> bool:
        """Execute wait action."""
        wait_time = 2.0  # Default wait time
        logger.info(f"Waiting {wait_time}s")
        time.sleep(wait_time)
        return True
    
    def _execute_back(self, step: Step, frame: Frame) -> bool:
        """Execute back key press."""
        logger.info("Pressing back key")
        
        # Log action
        if self.current_run_id:
            self.datastore.append_action(
                run_id=self.current_run_id,
                screen=self.last_screen,
                action="back",
                success=True
            )
        
        self.device.key_back()
        return True
    
    def _execute_assert(self, step: Step, frame: Frame) -> bool:
        """Execute assertion check."""
        if not step.target:
            logger.error("Assert step missing target")
            return False
        
        # Try to resolve target
        candidate = self.resolver.resolve(step.target, frame)
        success = candidate is not None
        
        logger.info(f"Assertion {'passed' if success else 'failed'}: {step.target}")
        return success
    
    def _verify_step_success(self, step: Step, original_frame: Frame) -> bool:
        """Verify that step was successful by capturing new frame.
        
        Args:
            step: Executed step
            original_frame: Frame before step execution
            
        Returns:
            True if step appears to have been successful
        """
        # Capture new frame
        new_frame = self.capture.grab()
        
        # Basic verification: check if frame changed
        # In a full implementation, you'd have more sophisticated checks
        frame_changed = self._frames_different(original_frame, new_frame)
        
        if frame_changed:
            logger.debug("Frame changed after step - likely successful")
            return True
        else:
            logger.warning("Frame did not change after step - may have failed")
            return False
    
    def _frames_different(self, frame1: Frame, frame2: Frame, threshold: float = 0.1) -> bool:
        """Check if two frames are significantly different.
        
        Args:
            frame1: First frame
            frame2: Second frame
            threshold: Difference threshold (0-1)
            
        Returns:
            True if frames are different
        """
        try:
            import cv2
            import numpy as np
            
            # Resize to same size if needed
            h1, w1 = frame1.image_bgr.shape[:2]
            h2, w2 = frame2.image_bgr.shape[:2]
            
            if (h1, w1) != (h2, w2):
                # Resize frame2 to match frame1
                frame2_resized = cv2.resize(frame2.image_bgr, (w1, h1))
            else:
                frame2_resized = frame2.image_bgr
            
            # Calculate difference
            diff = cv2.absdiff(frame1.image_bgr, frame2_resized)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Calculate percentage of changed pixels
            total_pixels = diff_gray.size
            changed_pixels = np.count_nonzero(diff_gray > 30)  # Threshold for significant change
            change_ratio = changed_pixels / total_pixels
            
            return change_ratio > threshold
            
        except Exception as e:
            logger.debug(f"Frame comparison failed: {e}")
            # If comparison fails, assume frames are different
            return True
    
    def recover_to_home(self) -> bool:
        """Attempt to recover to home screen.
        
        Returns:
            True if recovery successful
        """
        logger.info("Attempting recovery to home screen")
        
        max_backs = 3
        for i in range(max_backs):
            logger.info(f"Recovery attempt {i+1}: pressing back")
            self.device.key_back()
            time.sleep(2.0)
            
            # Check if we're home
            frame = self.capture.grab()
            ocr_results = self.ocr.text_in_roi(frame.image_bgr, (0, 0, 1, 1))
            screen = identify_screen(frame, ocr_results)
            
            if screen == "home":
                logger.info("Successfully recovered to home screen")
                return True
        
        # Try home key as last resort
        logger.info("Back recovery failed, trying home key")
        self.device.key_home()
        time.sleep(3.0)
        
        frame = self.capture.grab()
        ocr_results = self.ocr.text_in_roi(frame.image_bgr, (0, 0, 1, 1))
        screen = identify_screen(frame, ocr_results)
        
        if screen == "home":
            logger.info("Home key recovery successful")
            return True
        
        logger.error("Failed to recover to home screen")
        return False