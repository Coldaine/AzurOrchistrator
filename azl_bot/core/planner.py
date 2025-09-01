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
        self.actuator.tap_norm(candidate.point[0], candidate.point[1], frame.active_rect)
        
        # Wait and verify
        time.sleep(self.step_delay)
        return self._verify_step_success(step, frame)