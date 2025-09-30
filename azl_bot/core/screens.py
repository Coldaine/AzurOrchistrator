"""Screen definitions, detection helpers, and simple navigation aids.

This module provides a lightweight ScreenDetector used by tests to
identify the current screen from OCR tokens and to compute valid
transitions/recovery actions. Values and behaviors are aligned to the
unit tests in ``tests/test_state_machine.py``.
"""

from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from loguru import logger

from .capture import Frame


class ScreenState(Enum):
    """Known screen states in Azur Lane.

    Note: Enum values are uppercase strings to satisfy tests that compare
    ``{state.value for state in ScreenState}`` to an expected set.
    """

    UNKNOWN = "UNKNOWN"
    LOADING = "LOADING"
    HOME = "HOME"
    BATTLE = "BATTLE"
    COMMISSIONS = "COMMISSIONS"
    MAILBOX = "MAILBOX"
    MISSIONS = "MISSIONS"
    BUILD = "BUILD"
    DOCK = "DOCK"
    SHOP = "SHOP"
    ACADEMY = "ACADEMY"
    EXERCISE = "EXERCISE"
    COMBAT = "COMBAT"


class ScreenTransition:
    """Defines valid screen transitions."""
    
    TRANSITIONS: Dict[ScreenState, Set[ScreenState]] = {
        ScreenState.HOME: {
            ScreenState.COMMISSIONS,
            ScreenState.MAILBOX,
            ScreenState.MISSIONS,
            ScreenState.BUILD,
            ScreenState.DOCK,
            ScreenState.SHOP,
            ScreenState.ACADEMY,
            ScreenState.EXERCISE,
            ScreenState.COMBAT,
            ScreenState.LOADING,
        },
        ScreenState.COMMISSIONS: {ScreenState.HOME, ScreenState.LOADING},
        ScreenState.MAILBOX: {ScreenState.HOME, ScreenState.LOADING},
        ScreenState.MISSIONS: {ScreenState.HOME, ScreenState.LOADING},
        ScreenState.BUILD: {ScreenState.HOME, ScreenState.LOADING},
        ScreenState.DOCK: {ScreenState.HOME, ScreenState.LOADING},
        ScreenState.SHOP: {ScreenState.HOME, ScreenState.LOADING},
        ScreenState.ACADEMY: {ScreenState.HOME, ScreenState.LOADING},
        ScreenState.EXERCISE: {ScreenState.HOME, ScreenState.LOADING},
        ScreenState.COMBAT: {ScreenState.HOME, ScreenState.LOADING},
        ScreenState.BATTLE: {ScreenState.HOME, ScreenState.LOADING},
        ScreenState.LOADING: {
            ScreenState.HOME,
            ScreenState.COMMISSIONS,
            ScreenState.MAILBOX,
            ScreenState.MISSIONS,
            ScreenState.BUILD,
            ScreenState.DOCK,
            ScreenState.SHOP,
            ScreenState.ACADEMY,
            ScreenState.EXERCISE,
            ScreenState.COMBAT,
        },
        ScreenState.UNKNOWN: {ScreenState.HOME, ScreenState.LOADING},
    }
    
    @classmethod
    def can_transition(cls, from_state: ScreenState, to_state: ScreenState) -> bool:
        """Check if transition is valid."""
        return to_state in cls.TRANSITIONS.get(from_state, set())


class ScreenStateMachine:
    """Manages screen navigation state."""
    
    def __init__(self):
        self.current_state = ScreenState.UNKNOWN
        self.state_history: List[ScreenState] = []
        self.max_history = 10
    
    def update_state(self, detected_screen: str) -> ScreenState:
        """Update state based on detected screen."""
        # Map string to enum
        state_map = {
            "home": ScreenState.HOME,
            "commissions": ScreenState.COMMISSIONS,
            "mailbox": ScreenState.MAILBOX,
            "missions": ScreenState.MISSIONS,
            "battle": ScreenState.BATTLE,
            "combat": ScreenState.COMBAT,
            "dock": ScreenState.DOCK,
            "shop": ScreenState.SHOP,
            "build": ScreenState.BUILD,
            "academy": ScreenState.ACADEMY,
            "exercise": ScreenState.EXERCISE,
            "loading": ScreenState.LOADING,
            "unknown": ScreenState.UNKNOWN,
        }
        
        new_state = state_map.get(detected_screen, ScreenState.UNKNOWN)
        
        if new_state != self.current_state:
            self.state_history.append(self.current_state)
            if len(self.state_history) > self.max_history:
                self.state_history.pop(0)
            self.current_state = new_state
        
        return new_state
    
    def get_navigation_path(self, target: ScreenState) -> List[ScreenState]:
        """Get path to navigate to target screen."""
        # Simple BFS for path finding
        if self.current_state == target:
            return []
        
        # For now, always go through HOME
        if self.current_state != ScreenState.HOME:
            return [ScreenState.HOME, target]
        
        return [target]


class ScreenDetector:
    """Heuristic screen detector with simple OCR token scoring.

    Exposes a few attributes and methods expected by the tests:
      - previous_state: last identified state (ScreenState)
      - state_confidence: float confidence score in [0, 1]
      - identify_screen(frame, ocr_results=None) -> ScreenState
      - get_valid_transitions(state) -> set[ScreenState]
      - navigate_to_home() -> list[dict]
    """

    def __init__(self) -> None:
        self.previous_state: ScreenState = ScreenState.UNKNOWN
        self.state_confidence: float = 0.0
        # Keyword map for OCR detection
        self._keywords: Dict[ScreenState, Set[str]] = {
            ScreenState.HOME: {"home", "commissions", "missions", "mailbox"},
            ScreenState.COMMISSIONS: {"commission", "commissions", "urgent", "daily", "weekly"},
            ScreenState.MAILBOX: {"mail", "inbox", "collect"},
            ScreenState.MISSIONS: {"missions", "tasks", "daily"},
            ScreenState.BATTLE: {"battle", "sortie", "stage"},
            ScreenState.COMBAT: {"combat", "sortie"},
            ScreenState.DOCK: {"dock", "ships", "enhance"},
            ScreenState.SHOP: {"shop", "store", "purchase"},
            ScreenState.BUILD: {"build", "construct"},
            ScreenState.ACADEMY: {"academy", "classroom"},
            ScreenState.EXERCISE: {"exercise", "pvp"},
        }

    def identify_screen(self, frame: Frame, ocr_results: Optional[Sequence[Dict]] = None) -> ScreenState:
        # If no OCR data, decide whether to keep previous state (if confident)
        if not ocr_results:
            if self.state_confidence > 0.5:
                return self.previous_state
            self.previous_state = ScreenState.UNKNOWN
            self.state_confidence = 0.0
            return self.previous_state

        # Normalize tokens
        tokens: List[Tuple[str, float]] = []
        for item in ocr_results:
            text = item.get("text") if isinstance(item, dict) else None
            conf = float(item.get("conf", 0)) if isinstance(item, dict) else 0.0
            if isinstance(text, str):
                tokens.append((text.lower(), conf))

        if not tokens:
            self.previous_state = ScreenState.UNKNOWN
            self.state_confidence = 0.0
            return self.previous_state

        # Score matches per screen
        scores: Dict[ScreenState, float] = {state: 0.0 for state in self._keywords}
        matches: Dict[ScreenState, int] = {state: 0 for state in self._keywords}
        for t, c in tokens:
            for state, keys in self._keywords.items():
                if any(k in t for k in keys):
                    scores[state] += c if c > 0 else 0.5  # default low weight
                    matches[state] += 1

        # Special handling: if Home keywords show up with others, prefer the more specific screen
        # by selecting state with the highest matches, then highest score.
        best_state: ScreenState = ScreenState.UNKNOWN
        best_tuple: Tuple[int, float] = (0, 0.0)
        for state in self._keywords:
            cand = (matches[state], scores[state])
            if cand > best_tuple:
                best_tuple = cand
                best_state = state

        if best_state is ScreenState.UNKNOWN:
            # Fall back to UNKNOWN if nothing matched
            self.previous_state = ScreenState.UNKNOWN
            self.state_confidence = 0.0
            return self.previous_state

        # Confidence heuristic: scale by number of matches and average conf
        m, s = best_tuple
        avg_conf = s / max(1, m)
        self.state_confidence = max(0.0, min(1.0, 0.3 * m + 0.7 * (avg_conf)))
        # If best detected is HOME but other screens have higher matches, adjust (already handled by tuple ordering)
        self.previous_state = best_state
        return best_state

    def get_valid_transitions(self, state: ScreenState) -> Set[ScreenState]:
        return ScreenTransition.TRANSITIONS.get(state, {ScreenState.HOME, ScreenState.LOADING})

    def navigate_to_home(self) -> List[Dict[str, str]]:
        # Conservative recovery: three back presses then home tap
        actions = [
            {"action": "back", "rationale": "Attempt to exit nested menus (1/3)"},
            {"action": "back", "rationale": "Attempt to exit nested menus (2/3)"},
            {"action": "back", "rationale": "Attempt to exit nested menus (3/3)"},
            {"action": "home", "rationale": "Return to home screen"},
        ]
        return actions