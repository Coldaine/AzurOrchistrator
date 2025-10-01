"""
Planner No-LLM Fallback Mode Design

This file documents the intended extensions to azl_bot/core/planner.py.
Due to repository file corruption, the complete implementation cannot be integrated.

INTENDED CHANGES TO Planner CLASS:
===================================

1. Add no-LLM fallback mode support:

class Planner:
    def __init__(self, device, capture, resolver, ocr, llm, store, logger):
        # ... existing initialization ...
        self.llm = llm
        self._llm_available = True
        self._fallback_mode = False
        
    def _check_llm_availability(self) -> bool:
        '''Check if LLM is available.'''
        try:
            if not self.llm or not hasattr(self.llm, '_client'):
                return False
            return self.llm._client is not None
        except Exception as e:
            logger.error(f"LLM availability check failed: {e}")
            return False
    
    def enable_fallback_mode(self):
        '''Enable no-LLM fallback mode.'''
        self._fallback_mode = True
        logger.warning("Planner operating in NO-LLM FALLBACK MODE")
    
    def disable_fallback_mode(self):
        '''Disable fallback mode and resume normal LLM operation.'''
        self._fallback_mode = False
        logger.info("Planner resumed normal LLM mode")
    
    def _generate_fallback_plan(self, goal: Dict[str, Any], context: Dict[str, Any]) -> Plan:
        '''
        Generate a simple static plan when LLM is unavailable.
        
        Args:
            goal: Goal description
            context: Execution context
            
        Returns:
            Simple static Plan
        '''
        logger.warning(f"Generating fallback plan for goal: {goal}")
        
        # Extract goal action
        action = goal.get("action", "unknown")
        
        # Determine fallback strategy based on goal
        if action in ["open_commissions", "collect_commissions"]:
            # Simple plan: tap Commissions in bottom_bar
            return Plan(
                screen="unknown",
                steps=[
                    Step(
                        action="tap",
                        target=Target(
                            kind="text",
                            value="Commissions",
                            region_hint="bottom_bar"
                        ),
                        rationale="Fallback: Navigate to commissions"
                    )
                ],
                done=False
            )
        
        elif action in ["collect_mail", "open_mail"]:
            # Simple plan: tap Mail icon in top_bar
            return Plan(
                screen="unknown",
                steps=[
                    Step(
                        action="tap",
                        target=Target(
                            kind="icon",
                            value="mail_icon",
                            region_hint="top_bar"
                        ),
                        rationale="Fallback: Navigate to mail"
                    )
                ],
                done=False
            )
        
        elif action == "go_back":
            # Simple plan: press back
            return Plan(
                screen="unknown",
                steps=[
                    Step(
                        action="back",
                        rationale="Fallback: Go back"
                    )
                ],
                done=False
            )
        
        else:
            # Generic fallback: press back button
            logger.warning(f"No specific fallback for action '{action}', using generic back")
            return Plan(
                screen="unknown",
                steps=[
                    Step(
                        action="back",
                        rationale=f"Fallback: Unknown goal action '{action}'"
                    )
                ],
                done=False
            )
    
    def propose_plan(self, frame: Frame, goal: Dict[str, Any], context: Dict[str, Any]) -> Plan:
        '''
        Generate action plan - uses LLM if available, fallback if not.
        
        Args:
            frame: Current screen frame
            goal: Goal description
            context: Execution context
            
        Returns:
            Generated Plan (from LLM or fallback)
        '''
        # Check if we should use fallback mode
        if self._fallback_mode:
            logger.info("Using fallback mode (explicitly enabled)")
            return self._generate_fallback_plan(goal, context)
        
        # Try LLM first
        if self._llm_available and self._check_llm_availability():
            try:
                logger.debug("Attempting LLM plan generation")
                plan = self.llm.propose_plan(frame, goal, context)
                logger.info(f"LLM plan generated: {plan.screen}, {len(plan.steps)} steps")
                return plan
                
            except Exception as e:
                logger.error(f"LLM plan generation failed: {e}")
                self._llm_available = False  # Mark as unavailable
                logger.warning("LLM marked unavailable, switching to fallback mode")
        
        # Fall back to static plan
        logger.warning("Using fallback plan (LLM unavailable)")
        return self._generate_fallback_plan(goal, context)
    
    def run_task(self, task: Task):
        '''Execute a task with automatic fallback support.'''
        logger.info(f"Starting task: {task.name}")
        
        # Check LLM availability at start
        if not self._check_llm_availability():
            logger.warning(f"LLM not available for task {task.name}, using fallback mode")
            self._fallback_mode = True
        
        # ... rest of existing run_task implementation ...
        
        max_iterations = 50
        iteration = 0
        
        while iteration < max_iterations:
            frame = self.capture.grab()
            
            # Check task success
            if task.success(frame, self.context):
                logger.info(f"Task {task.name} completed")
                task.on_success(self, frame)
                break
            
            # Get plan (with automatic fallback)
            goal = task.goal()
            plan = self.propose_plan(frame, goal, self.context)
            
            # Log if using fallback
            if self._fallback_mode or not self._llm_available:
                logger.info(f"Executing fallback plan: {plan.steps}")
            
            # Execute plan steps
            for step in plan.steps:
                success = self.run_step(step, frame)
                if not success:
                    logger.warning(f"Step failed: {step}")
                    self.recover()
                    break
                
                time.sleep(1.0)
                frame = self.capture.grab()
            
            iteration += 1
        
        if iteration >= max_iterations:
            logger.error(f"Task {task.name} exceeded max iterations")


USAGE EXAMPLE:
==============

# Normal operation - LLM will be used if available
planner = Planner(device, capture, resolver, ocr, llm, store, logger)
planner.run_task(currencies_task)

# Explicitly force fallback mode (for testing or when LLM is known to be down)
planner.enable_fallback_mode()
planner.run_task(currencies_task)
planner.disable_fallback_mode()

# Automatic fallback - if LLM fails mid-execution
# The planner will automatically switch to fallback mode


FALLBACK PLAN CHARACTERISTICS:
===============================

1. Simple, deterministic actions based on common game UI patterns
2. Uses text/icon selectors with region hints (bottom_bar, top_bar, etc.)
3. Always returns valid Plan objects (never None)
4. Logs clearly when fallback mode is active
5. Can be extended with more goal-specific fallback strategies

LOGGING OUTPUT:
===============

When fallback mode is active:
  WARNING: Planner operating in NO-LLM FALLBACK MODE
  WARNING: Using fallback plan (LLM unavailable)
  INFO: Executing fallback plan: [Step(action='tap', ...)]

This makes it clear to operators that the system is running with reduced intelligence
but still attempting to achieve goals through simple static plans.


INTEGRATION WITH StateLoop:
============================

The StateLoop can use the planner in either mode:

# In a task execution context:
from azl_bot.core.loop import StateLoop, LoopConfig

loop_config = LoopConfig(max_retries=3, recovery_enabled=True)
state_loop = StateLoop(loop_config, capture, actuator, device)

# Get plan from planner (automatically uses fallback if needed)
plan = planner.propose_plan(frame, goal, context)

# Execute each step with verification and recovery
for step in plan.steps:
    def execute_step():
        planner.run_step(step, frame)
    
    success, result_frame = state_loop.run_action_with_recovery(
        execute_step,
        postcondition=lambda f: verify_step_success(step, f)
    )
    
    if not success:
        logger.error(f"Step {step} failed even with recovery")
        break
"""
