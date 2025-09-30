# Task Registry and Daily Maintenance Guide

This document describes the new task registry system and the daily_maintenance task.

## Task Registry

The task registry provides a centralized way to discover, manage, and execute bot tasks.

### Features

- **Task Discovery**: List all available tasks with descriptions
- **Metadata Support**: Each task can have a description and expected duration
- **Easy Registration**: Simple decorator-based registration
- **CLI Integration**: Execute tasks via `python -m azl_bot.tasks`

### Usage

#### Listing Available Tasks

```bash
# Using the CLI module
python -m azl_bot.tasks --list

# Using the shell script
./scripts/run_task.sh --list
```

Example output:
```
Available tasks:
================
  commissions          - Read commission information (expected: 30s)
  currencies           - Read currency balances from top bar (expected: 10s)
  daily_maintenance    - Complete daily maintenance routine (expected: 120s)
  pickups              - Collect main menu pickups (mail, missions, etc.) (expected: 60s)
```

#### Running a Task

```bash
# Using the CLI module
python -m azl_bot.tasks <task_name>

# Using the shell script
./scripts/run_task.sh <task_name>
```

Examples:
```bash
python -m azl_bot.tasks currencies
python -m azl_bot.tasks daily_maintenance
./scripts/run_task.sh pickups
```

#### Programmatic Access

```python
from azl_bot.tasks import get_task, list_tasks, has_task

# List all tasks
tasks = list_tasks()
for name, metadata in tasks.items():
    print(f"{name}: {metadata.description}")

# Check if a task exists
if has_task("daily_maintenance"):
    task = get_task("daily_maintenance")
    # Use the task...
```

### Registering New Tasks

To add a new task to the registry:

```python
from azl_bot.tasks import register

@register(
    "my_task",
    description="My custom task",
    expected_duration=60  # seconds
)
def create_my_task():
    return MyTaskClass()
```

Then update `azl_bot/tasks/__init__.py` to import your task factory.

## Daily Maintenance Task

The `daily_maintenance` task is an orchestrated workflow that executes multiple daily activities in sequence.

### What It Does

1. **Go Home**: Navigate to the home screen
2. **Collect Mailbox**: Collect any available mail and mission rewards
3. **Check Commissions**: Read commission status and collect ready ones
4. **Record Currencies**: Record current Oil, Coins, and Gems balances

### Idempotent Design

The task is designed to be **idempotent** - safe to re-run multiple times:

- Already-claimed rewards are detected and reported as `already_complete`
- No-ops don't generate errors
- Each step returns a clear status (success, already_complete, failed, skipped)

### Running Daily Maintenance

```bash
# CLI
python -m azl_bot.tasks daily_maintenance

# Shell script
./scripts/run_task.sh daily_maintenance
```

### Output

The task produces:

1. **Console Logs**: Step-by-step progress with status and confidence
2. **Summary JSON**: Detailed results written to `~/.azlbot/summaries/`

Example summary JSON:
```json
{
  "task": "daily_maintenance",
  "timestamp": "2024-09-30T15:30:00",
  "success": true,
  "duration": 95.3,
  "steps": [
    {
      "name": "go_home",
      "status": "already_complete",
      "confidence": 0.9,
      "details": "Already on home screen",
      "duration": 0.5
    },
    {
      "name": "collect_mailbox",
      "status": "success",
      "confidence": 0.85,
      "details": "Collected 3 pickups",
      "duration": 45.2
    },
    {
      "name": "check_commissions",
      "status": "success",
      "confidence": 0.85,
      "details": "Commission status checked",
      "duration": 30.1
    },
    {
      "name": "record_currencies",
      "status": "success",
      "confidence": 0.85,
      "details": "Currencies recorded",
      "duration": 19.5
    }
  ]
}
```

### Re-running

If you re-run the task within a short time:

```bash
python -m azl_bot.tasks daily_maintenance
```

The task will detect already-completed steps and report:
- `already_complete` for items already claimed
- `success` for fresh data collection
- No errors or duplicate actions

### Step Results

Each step returns a `StepResult` with:
- `step_name`: Name of the step
- `status`: One of: `success`, `already_complete`, `failed`, `skipped`
- `confidence`: Float 0-1 indicating confidence in the result
- `details`: Human-readable details
- `duration`: Time taken in seconds

## CLI Interface

The new CLI interface (`azl_bot/tasks/__main__.py`) provides:

### Commands

```bash
# List tasks
python -m azl_bot.tasks --list
python -m azl_bot.tasks -l
python -m azl_bot.tasks list

# Show help
python -m azl_bot.tasks --help
python -m azl_bot.tasks -h
python -m azl_bot.tasks help

# Run a task
python -m azl_bot.tasks <task_name>
```

### Summary Files

After each task run, a summary JSON is written to:
```
~/.azlbot/summaries/<task_name>_<timestamp>.json
```

This allows for:
- Reliability tracking
- Performance monitoring
- Historical analysis
- Debugging

## Integration with Existing Code

### Bootstrap

The bootstrap system now includes all registered tasks:

```python
from azl_bot.core.bootstrap import bootstrap_from_config

components = bootstrap_from_config("config/app.yaml")
tasks = components["tasks"]  # Dict of all tasks including daily_maintenance
```

### UI

The GUI now includes the daily_maintenance task in the dropdown:
- Open the GUI: `python -m azl_bot.ui.app`
- Select "daily_maintenance" from the task dropdown
- Click "Start" to run

### Scripts

The shell script `scripts/run_task.sh` now uses the registry:
```bash
./scripts/run_task.sh daily_maintenance
```

## Testing

### Registry Tests

Run the registry unit tests:
```bash
python tests/test_registry.py
```

### Syntax Validation

Validate all task files:
```bash
python tests/basic_test.py
```

## Architecture

```
azl_bot/
├── tasks/
│   ├── __init__.py           # Registry setup and exports
│   ├── __main__.py           # CLI entry point
│   ├── registry.py           # Task registry implementation
│   ├── daily.py              # Daily maintenance task
│   ├── currencies.py         # Currency reading task
│   ├── pickups.py            # Pickup collection task
│   └── commissions.py        # Commission reading task
└── core/
    └── bootstrap.py          # Component initialization

scripts/
└── run_task.sh               # Shell script wrapper

tests/
└── test_registry.py          # Registry unit tests
```

## Future Enhancements

Possible improvements:

1. **Task Dependencies**: Support for task prerequisites
2. **Scheduling**: Cron-like task scheduling
3. **Parallel Execution**: Run independent tasks concurrently
4. **Progress Callbacks**: Real-time progress reporting
5. **Task Chains**: Define complex workflows
6. **Retry Logic**: Automatic retry on transient failures
7. **Notification System**: Alert on completion or errors

## Troubleshooting

### Task Not Found

If you get "Task not found" errors:
1. Check the task name with `--list`
2. Ensure the task is imported in `azl_bot/tasks/__init__.py`
3. Verify the task factory is registered

### Re-run Errors

If re-running produces errors instead of graceful no-ops:
1. Check the task's idempotency logic
2. Review the step result statuses in the summary JSON
3. Examine logs for specific failure points

### Configuration Issues

If the task can't find your config:
1. Ensure `config/app.yaml` exists
2. Set `AZL_CONFIG` environment variable if needed
3. Check config file permissions

## Summary

The task registry and daily_maintenance task provide:

✓ Unified task discovery and execution
✓ Idempotent daily maintenance workflow
✓ Per-run summaries for reliability tracking
✓ CLI and script integration
✓ Easy extensibility for new tasks

For more details, see the implementation in `azl_bot/tasks/` directory.
