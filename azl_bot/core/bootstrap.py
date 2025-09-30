"""Bootstrap and initialization for Azur Lane Bot."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from .actuator import Actuator
from .capture import Capture
from .configs import AppConfig, load_config, create_default_config
from .datastore import DataStore
from .device import Device
from .llm_client import LLMClient
from .ocr import OCRClient
from .planner import Planner
from .resolver import Resolver
from .screens import ScreenStateMachine


def bootstrap_from_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Bootstrap all components from configuration.
    
    Args:
        config_path: Path to config file, or None to use default
        
    Returns:
        Dictionary of initialized components
    """
    # Load configuration
    if config_path:
        config = load_config(config_path)
    else:
        config_path = os.getenv("AZL_CONFIG", "./config/app.yaml")
        if Path(config_path).exists():
            config = load_config(config_path)
        else:
            logger.warning(f"Config not found at {config_path}, using defaults")
            config = create_default_config()
    
    logger.info(f"Loaded configuration")
    
    # Initialize device
    device = Device(
        adb_serial=config.emulator.adb_serial,
        package_name=config.emulator.package_name
    )
    
    # Initialize capture
    capture = Capture(device)
    
    # Initialize OCR
    ocr = OCRClient(
        backend=config.resolver.ocr,
        config=config.resolver.thresholds.model_dump()
    )
    
    # Initialize LLM
    llm = None
    try:
        api_key = config.llm_api_key
        llm = LLMClient(
            provider=config.llm.provider,
            model=config.llm.model,
            api_key=api_key,
            endpoint=config.llm.endpoint,
            max_tokens=config.llm.max_tokens,
            temperature=config.llm.temperature
        )
    except ValueError as e:
        logger.warning(f"LLM not available: {e}")
    
    # Initialize resolver
    templates_dir = str(Path("./config/templates").absolute())
    resolver = Resolver(
        config=config.resolver.model_dump(),
        ocr_client=ocr,
        templates_dir=templates_dir,
        llm=llm
    )
    
    # Initialize datastore
    data_dir = config.data_dir
    datastore = DataStore(base_dir=data_dir)
    
    # Initialize actuator
    actuator = Actuator(device=device, capture=capture)
    
    # Initialize planner
    planner = Planner(
        llm=llm,
        resolver=resolver,
        actuator=actuator,
        datastore=datastore,
        config=config
    )
    
    # Initialize screen state machine
    screen_state_machine = ScreenStateMachine()
    
    # Initialize tasks
    from azl_bot.tasks import currencies, pickups, commissions
    tasks = {
        "currencies": currencies,
        "pickups": pickups,
        "commissions": commissions
    }
    
    components = {
        "config": config,
        "device": device,
        "capture": capture,
        "ocr": ocr,
        "llm": llm,
        "resolver": resolver,
        "datastore": datastore,
        "actuator": actuator,
        "planner": planner,
        "tasks": tasks,
        "screen_state_machine": screen_state_machine,
        "hasher": capture.hasher
    }
    
    return components


def test_components() -> bool:
    """Test basic component initialization.
    
    Returns:
        True if all components initialize successfully
    """
    try:
        components = bootstrap_from_config()
        logger.info("All components initialized successfully")
        logger.info(f"Components: {list(components.keys())}")
        return True
    except Exception as e:
        logger.error(f"Component initialization failed: {e}")
        return False


def run_task_cli(task_name: str) -> None:
    """Run a task from CLI.
    
    Args:
        task_name: Name of task to run (currencies, pickups, commissions)
    """
    components = bootstrap_from_config()
    
    if task_name not in components["tasks"]:
        logger.error(f"Unknown task: {task_name}")
        logger.info(f"Available tasks: {list(components['tasks'].keys())}")
        return
    
    task_module = components["tasks"][task_name]
    planner = components["planner"]
    
    logger.info(f"Running task: {task_name}")
    
    # Get task goal
    goal = task_module.goal()
    logger.info(f"Task goal: {goal}")
    
    # Execute task
    try:
        result = planner.execute_goal(goal)
        logger.info(f"Task completed: {result}")
    except Exception as e:
        logger.error(f"Task failed: {e}")
