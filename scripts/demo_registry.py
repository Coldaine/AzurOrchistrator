#!/usr/bin/env python3
"""
Demonstration of the task registry and daily maintenance features.

This script shows the key functionality without requiring a full environment setup.
"""

import sys
from pathlib import Path

# Ensure we can import from the project
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("Task Registry & Daily Maintenance - Feature Demonstration")
print("=" * 70)

# Part 1: Registry Core Functionality
print("\n" + "─" * 70)
print("1. Task Registry Core Functionality")
print("─" * 70)

# Import registry module directly to avoid triggering full task imports
from azl_bot.tasks.registry import TaskRegistry, TaskMetadata

# Create a registry
registry = TaskRegistry()

# Register example tasks
@registry.register("example_task", description="Example task", expected_duration=30)
def create_example():
    return {"name": "example_task", "ready": True}

@registry.register("another_task", description="Another example", expected_duration=60)
def create_another():
    return {"name": "another_task", "ready": True}

# Demonstrate registry features
print("\n✓ Task Registration:")
print(f"  Registered {len(registry.list_tasks())} tasks")

print("\n✓ Task Listing:")
for name, metadata in registry.list_tasks().items():
    duration_str = f" ({metadata.expected_duration}s)" if metadata.expected_duration else ""
    print(f"  • {name:20s} - {metadata.description}{duration_str}")

print("\n✓ Task Checking:")
print(f"  has_task('example_task'): {registry.has_task('example_task')}")
print(f"  has_task('nonexistent'):  {registry.has_task('nonexistent')}")

print("\n✓ Task Retrieval:")
task = registry.get_task('example_task')
print(f"  get_task('example_task'): {task}")

# Part 2: Step Result Structure
print("\n" + "─" * 70)
print("2. Daily Maintenance Step Result Structure")
print("─" * 70)

# Manually define the key classes to avoid import issues
from dataclasses import dataclass
from enum import Enum

class StepStatus(str, Enum):
    SUCCESS = "success"
    ALREADY_COMPLETE = "already_complete"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class StepResult:
    step_name: str
    status: StepStatus
    confidence: float
    details: str = None
    duration: float = 0.0

# Create example results
results = [
    StepResult(
        step_name="go_home",
        status=StepStatus.ALREADY_COMPLETE,
        confidence=0.9,
        details="Already on home screen",
        duration=0.5
    ),
    StepResult(
        step_name="collect_mailbox",
        status=StepStatus.SUCCESS,
        confidence=0.85,
        details="Collected 3 pickups",
        duration=45.2
    ),
    StepResult(
        step_name="check_commissions",
        status=StepStatus.SUCCESS,
        confidence=0.85,
        details="Commission status checked",
        duration=30.1
    ),
    StepResult(
        step_name="record_currencies",
        status=StepStatus.SUCCESS,
        confidence=0.85,
        details="Currencies recorded",
        duration=19.5
    ),
]

print("\n✓ Example daily_maintenance execution:")
for result in results:
    status_icon = {
        StepStatus.SUCCESS: "✓",
        StepStatus.ALREADY_COMPLETE: "↻",
        StepStatus.FAILED: "✗",
        StepStatus.SKIPPED: "⊘"
    }.get(result.status, "?")
    
    print(f"  {status_icon} {result.step_name:20s} │ {result.status.value:18s} │ "
          f"conf: {result.confidence:.2f} │ {result.duration:5.1f}s")
    if result.details:
        print(f"    └─ {result.details}")

total_duration = sum(r.duration for r in results)
success_count = sum(1 for r in results if r.status in (StepStatus.SUCCESS, StepStatus.ALREADY_COMPLETE))
print(f"\n  Total: {total_duration:.1f}s | Successful: {success_count}/{len(results)} steps")

# Part 3: Summary JSON Example
print("\n" + "─" * 70)
print("3. Summary JSON Output Example")
print("─" * 70)

import json
from datetime import datetime

summary = {
    "task": "daily_maintenance",
    "timestamp": datetime.now().isoformat(),
    "success": True,
    "duration": total_duration,
    "steps": [
        {
            "name": r.step_name,
            "status": r.status.value,
            "confidence": r.confidence,
            "details": r.details,
            "duration": r.duration,
        }
        for r in results
    ]
}

print("\n✓ Example summary JSON:")
print(json.dumps(summary, indent=2))

# Part 4: Implementation Summary
print("\n" + "─" * 70)
print("4. Implementation Summary")
print("─" * 70)

files_info = [
    ("azl_bot/tasks/registry.py", "Task registry system with metadata"),
    ("azl_bot/tasks/__init__.py", "Registry setup and task registration"),
    ("azl_bot/tasks/__main__.py", "CLI entry point for task execution"),
    ("azl_bot/tasks/daily.py", "Idempotent daily maintenance workflow"),
    ("azl_bot/core/bootstrap.py", "Component initialization system"),
    ("scripts/run_task.sh", "Shell script wrapper for tasks"),
    ("docs/TASK_REGISTRY.md", "Complete documentation"),
]

print("\n✓ Implementation files:")
for filepath, description in files_info:
    path = Path(__file__).parent.parent / filepath
    if path.exists():
        lines = len(path.read_text().splitlines())
        size_kb = path.stat().st_size / 1024
        print(f"  • {filepath:35s} │ {lines:4d} lines │ {size_kb:6.1f} KB")
        print(f"    └─ {description}")
    else:
        print(f"  ✗ {filepath:35s} │ NOT FOUND")

# Part 5: Usage Examples
print("\n" + "─" * 70)
print("5. Usage Examples")
print("─" * 70)

print("\n✓ CLI Commands:")
print("  # List all available tasks")
print("  $ python -m azl_bot.tasks --list")
print()
print("  # Run daily maintenance")
print("  $ python -m azl_bot.tasks daily_maintenance")
print()
print("  # Or using the shell script")
print("  $ ./scripts/run_task.sh daily_maintenance")

print("\n✓ Programmatic Usage:")
print("""
  from azl_bot.tasks import get_task, list_tasks
  
  # List tasks
  for name, metadata in list_tasks().items():
      print(f"{name}: {metadata.description}")
  
  # Run a task
  task = get_task("daily_maintenance")
  results = task.execute(planner)
""")

# Final Summary
print("\n" + "=" * 70)
print("Key Features Implemented")
print("=" * 70)

features = [
    "✓ Task registry with decorator-based registration",
    "✓ Metadata support (description, expected_duration)",
    "✓ Idempotent daily_maintenance workflow",
    "✓ Step-level results with status tracking",
    "✓ CLI interface: python -m azl_bot.tasks",
    "✓ Shell script integration",
    "✓ JSON summary output for reliability tracking",
    "✓ Graceful handling of already-complete states",
    "✓ Safe re-runs without errors",
]

for feature in features:
    print(f"  {feature}")

print("\n" + "=" * 70)
print("See docs/TASK_REGISTRY.md for complete documentation")
print("=" * 70)
