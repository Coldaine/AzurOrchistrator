"""CLI entry point for task execution.

Usage:
    python -m azl_bot.tasks                    # List available tasks
    python -m azl_bot.tasks <task_name>        # Run a specific task
    python -m azl_bot.tasks daily_maintenance  # Run daily maintenance
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from ..core.bootstrap import bootstrap_from_config, bootstrap_from_config_object
from ..core.configs import create_default_config, load_config
from . import get_task, list_tasks


def print_tasks():
    """Print all available tasks."""
    tasks = list_tasks()
    
    print("\nAvailable tasks:")
    print("================")
    for name, metadata in sorted(tasks.items()):
        duration_str = f" (expected: {metadata.expected_duration}s)" if metadata.expected_duration else ""
        print(f"  {name:20s} - {metadata.description}{duration_str}")
    print()


def write_summary(task_name: str, success: bool, results: Optional[list] = None, duration: float = 0.0) -> None:
    """Write task execution summary to JSON file.
    
    Args:
        task_name: Name of the executed task
        success: Whether task succeeded
        results: Optional list of step results (for daily_maintenance)
        duration: Total execution duration in seconds
    """
    timestamp = datetime.now().isoformat()
    
    summary = {
        "task": task_name,
        "timestamp": timestamp,
        "success": success,
        "duration": duration,
    }
    
    # Add step results if available (for daily_maintenance)
    if results:
        summary["steps"] = [
            {
                "name": r.step_name,
                "status": r.status.value,
                "confidence": r.confidence,
                "details": r.details,
                "duration": r.duration,
            }
            for r in results
        ]
    
    # Write to data directory
    try:
        # Try to get data directory from config
        config_paths = ["config/app.yaml", Path.home() / ".azlbot" / "config.yaml"]
        data_dir = Path.home() / ".azlbot"  # Default
        
        for config_path in config_paths:
            if Path(config_path).exists():
                try:
                    config = load_config(config_path)
                    data_dir = config.data_dir
                    break
                except Exception:
                    pass
        
        # Create summaries directory
        summaries_dir = data_dir / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        
        # Write summary file
        summary_file = summaries_dir / f"{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary written to: {summary_file}")
        
    except Exception as e:
        logger.warning(f"Could not write summary file: {e}")
        # Still print summary to stdout
        print("\nTask Summary:")
        print("=============")
        print(json.dumps(summary, indent=2))


def run_task_cli(task_name: str) -> bool:
    """Run a task from the CLI.
    
    Args:
        task_name: Name of the task to run
        
    Returns:
        True if task succeeded
    """
    start_time = time.time()
    
    try:
        # Load configuration
        config_path = Path("config/app.yaml")
        if not config_path.exists():
            logger.warning("No config file found, using defaults")
            config = create_default_config()
            components = bootstrap_from_config_object(config)
        else:
            components = bootstrap_from_config(config_path)
        
        planner = components["planner"]
        
        # Get task from registry
        logger.info(f"Running task: {task_name}")
        task = get_task(task_name)
        
        # Special handling for daily_maintenance
        if task_name == "daily_maintenance":
            results = task.execute(planner)
            success = all(
                r.status in ("success", "already_complete")
                for r in results
            )
            duration = time.time() - start_time
            
            # Write summary with step details
            write_summary(task_name, success, results, duration)
            
            return success
        else:
            # Regular task execution
            success = planner.run_task(task)
            duration = time.time() - start_time
            
            # Write summary
            write_summary(task_name, success, None, duration)
            
            return success
            
    except KeyError as e:
        logger.error(f"Task not found: {e}")
        print_tasks()
        return False
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        # No arguments - list tasks
        print_tasks()
        sys.exit(0)
    
    task_name = sys.argv[1]
    
    # Handle special commands
    if task_name in ("--help", "-h", "help"):
        print(__doc__)
        print_tasks()
        sys.exit(0)
    
    if task_name in ("--list", "-l", "list"):
        print_tasks()
        sys.exit(0)
    
    # Run the task
    success = run_task_cli(task_name)
    
    if success:
        logger.info(f"✓ Task '{task_name}' completed successfully")
        sys.exit(0)
    else:
        logger.error(f"✗ Task '{task_name}' failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
