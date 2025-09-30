"""Task modules for specific game objectives."""

# Import registry functions
from .registry import get_task, has_task, list_tasks, register

# Register existing tasks
from .commissions import create_commissions_task
from .currencies import create_currencies_task
from .daily import create_daily_task
from .pickups import create_pickups_task

# Register the built-in tasks
_registry_setup = False

def _setup_registry():
    """Register built-in tasks."""
    global _registry_setup
    if _registry_setup:
        return
    
    register("commissions", description="Read commission information", expected_duration=30)(create_commissions_task)
    register("currencies", description="Read currency balances from top bar", expected_duration=10)(create_currencies_task)
    register("pickups", description="Collect main menu pickups (mail, missions, etc.)", expected_duration=60)(create_pickups_task)
    register("daily_maintenance", description="Complete daily maintenance routine", expected_duration=120)(create_daily_task)
    
    _registry_setup = True

# Setup on import
_setup_registry()

__all__ = ["get_task", "has_task", "list_tasks", "register"]