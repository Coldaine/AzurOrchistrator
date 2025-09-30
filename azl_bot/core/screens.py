"""Screen definitions and region helpers."""

from enum import Enum
from typing import List, Tuple, Dict, Set

import cv2
import numpy as np
from loguru import logger

from .capture import Frame


class ScreenState(Enum):
    """Known screen states in Azur Lane."""
    UNKNOWN = "unknown"
    LOADING = "loading"
    HOME = "home"
    BATTLE = "battle"
    COMMISSIONS = "commissions"
    DORM = "dorm"
    DOCK = "dock"
    SHOP = "shop"
    BUILD = "build"
    ACADEMY = "academy"
    RESEARCH = "research"
    COLLECTION = "collection"


class ScreenTransition:
    """Defines valid screen transitions."""
    
    TRANSITIONS: Dict[ScreenState, Set[ScreenState]] = {
        ScreenState.HOME: {
            ScreenState.BATTLE, ScreenState.COMMISSIONS, 
            ScreenState.DORM, ScreenState.DOCK, ScreenState.SHOP,
            ScreenState.BUILD, ScreenState.ACADEMY, ScreenState.RESEARCH
        },
        ScreenState.COMMISSIONS: {ScreenState.HOME},
        ScreenState.BATTLE: {ScreenState.HOME},
        ScreenState.DORM: {ScreenState.HOME},
        ScreenState.DOCK: {ScreenState.HOME},
        ScreenState.SHOP: {ScreenState.HOME},
        ScreenState.BUILD: {ScreenState.HOME},
        ScreenState.ACADEMY: {ScreenState.HOME},
        ScreenState.RESEARCH: {ScreenState.HOME},
        ScreenState.COLLECTION: {ScreenState.HOME},
        ScreenState.LOADING: {ScreenState.HOME, ScreenState.UNKNOWN}
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
            "battle": ScreenState.BATTLE,
            "dorm": ScreenState.DORM,
            "dock": ScreenState.DOCK,
            "shop": ScreenState.SHOP,
            "build": ScreenState.BUILD,
            "academy": ScreenState.ACADEMY,
            "research": ScreenState.RESEARCH,
            "collection": ScreenState.COLLECTION,
            "loading": ScreenState.LOADING,
            "unknown": ScreenState.UNKNOWN
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