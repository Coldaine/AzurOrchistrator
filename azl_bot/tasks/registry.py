"""Task registry for discovering and managing bot tasks."""

from typing import Any, Callable, Dict, Optional, Protocol

from loguru import logger


class TaskMetadata:
    """Metadata for a registered task."""
    
    def __init__(
        self,
        name: str,
        factory: Callable,
        description: Optional[str] = None,
        expected_duration: Optional[int] = None
    ):
        """Initialize task metadata.
        
        Args:
            name: Unique task identifier
            factory: Callable that creates task instance
            description: Human-readable task description
            expected_duration: Expected duration in seconds
        """
        self.name = name
        self.factory = factory
        self.description = description or f"Task: {name}"
        self.expected_duration = expected_duration


class TaskRegistry:
    """Registry for bot tasks."""
    
    def __init__(self):
        """Initialize empty task registry."""
        self._tasks: Dict[str, TaskMetadata] = {}
    
    def register(
        self,
        name: str,
        description: Optional[str] = None,
        expected_duration: Optional[int] = None
    ) -> Callable:
        """Decorator to register a task factory function.
        
        Args:
            name: Unique task identifier
            description: Optional task description
            expected_duration: Optional expected duration in seconds
            
        Returns:
            Decorator function
            
        Example:
            @register("my_task", description="Does something", expected_duration=60)
            def create_my_task():
                return MyTask()
        """
        def decorator(factory: Callable) -> Callable:
            if name in self._tasks:
                logger.warning(f"Task '{name}' is already registered, overwriting")
            
            self._tasks[name] = TaskMetadata(
                name=name,
                factory=factory,
                description=description,
                expected_duration=expected_duration
            )
            
            logger.debug(f"Registered task: {name}")
            return factory
        
        return decorator
    
    def get_task(self, name: str) -> Any:
        """Get a task instance by name.
        
        Args:
            name: Task name
            
        Returns:
            Task instance created by the factory
            
        Raises:
            KeyError: If task name is not registered
        """
        if name not in self._tasks:
            raise KeyError(f"Task '{name}' not found in registry. Available: {list(self._tasks.keys())}")
        
        metadata = self._tasks[name]
        return metadata.factory()
    
    def list_tasks(self) -> Dict[str, TaskMetadata]:
        """List all registered tasks.
        
        Returns:
            Dictionary mapping task names to metadata
        """
        return self._tasks.copy()
    
    def has_task(self, name: str) -> bool:
        """Check if a task is registered.
        
        Args:
            name: Task name
            
        Returns:
            True if task is registered
        """
        return name in self._tasks


# Global task registry instance
_registry = TaskRegistry()


def register(
    name: str,
    description: Optional[str] = None,
    expected_duration: Optional[int] = None
) -> Callable:
    """Register a task factory function.
    
    This is a convenience function that uses the global registry.
    
    Args:
        name: Unique task identifier
        description: Optional task description
        expected_duration: Optional expected duration in seconds
        
    Returns:
        Decorator function
    """
    return _registry.register(name, description, expected_duration)


def get_task(name: str) -> Any:
    """Get a task instance by name from the global registry.
    
    Args:
        name: Task name
        
    Returns:
        Task instance
    """
    return _registry.get_task(name)


def list_tasks() -> Dict[str, TaskMetadata]:
    """List all registered tasks from the global registry.
    
    Returns:
        Dictionary mapping task names to metadata
    """
    return _registry.list_tasks()


def has_task(name: str) -> bool:
    """Check if a task is registered in the global registry.
    
    Args:
        name: Task name
        
    Returns:
        True if task is registered
    """
    return _registry.has_task(name)
