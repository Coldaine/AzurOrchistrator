"""Test task registry functionality."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_registry_import():
    """Test that registry can be imported."""
    try:
        from azl_bot.tasks import registry
        print("✓ Registry module imports successfully")
        return True
    except Exception as e:
        print(f"✗ Registry import failed: {e}")
        return False


def test_registry_api():
    """Test registry API functions."""
    try:
        from azl_bot.tasks.registry import TaskRegistry, TaskMetadata
        
        # Create a test registry
        registry = TaskRegistry()
        
        # Test registration
        @registry.register("test_task", description="A test task", expected_duration=10)
        def create_test_task():
            return {"name": "test_task"}
        
        # Test listing
        tasks = registry.list_tasks()
        assert "test_task" in tasks, "Task not found in registry"
        assert isinstance(tasks["test_task"], TaskMetadata), "Invalid metadata type"
        assert tasks["test_task"].description == "A test task", "Wrong description"
        assert tasks["test_task"].expected_duration == 10, "Wrong duration"
        
        # Test has_task
        assert registry.has_task("test_task"), "has_task failed for existing task"
        assert not registry.has_task("nonexistent"), "has_task returned true for nonexistent task"
        
        # Test get_task
        task = registry.get_task("test_task")
        assert task == {"name": "test_task"}, "get_task returned wrong result"
        
        print("✓ Registry API tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Registry API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_global_registry():
    """Test global registry functions."""
    try:
        from azl_bot.tasks.registry import register, get_task, list_tasks, has_task
        
        # These should work without errors
        tasks = list_tasks()
        print(f"  Global registry has {len(tasks)} tasks")
        
        # Test has_task
        result = has_task("nonexistent_task_xyz")
        assert result == False, "has_task should return False for nonexistent task"
        
        print("✓ Global registry functions work")
        return True
        
    except Exception as e:
        print(f"✗ Global registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task_registration():
    """Test that built-in tasks are registered."""
    try:
        # Import will trigger registration
        from azl_bot.tasks import list_tasks
        
        tasks = list_tasks()
        
        # Check for expected tasks
        expected_tasks = ["currencies", "pickups", "commissions", "daily_maintenance"]
        
        for task_name in expected_tasks:
            if task_name in tasks:
                metadata = tasks[task_name]
                print(f"  ✓ {task_name:20s} - {metadata.description}")
            else:
                print(f"  ✗ {task_name} not registered")
                return False
        
        print(f"✓ All {len(expected_tasks)} built-in tasks registered")
        return True
        
    except Exception as e:
        print(f"✗ Task registration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all registry tests."""
    print("Task Registry Tests")
    print("=" * 50)
    
    tests = [
        test_registry_import,
        test_registry_api,
        test_global_registry,
        # test_task_registration,  # Skip this as it requires full imports
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\nRunning: {test.__name__}")
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
