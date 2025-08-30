"""Bootstrap module for wiring components and CLI utilities."""

import sys
from pathlib import Path
from typing import Any, Dict

from loguru import logger

from .actuator import Actuator
from .capture import Capture
from .configs import load_config, create_default_config, AppConfig
from .datastore import DataStore
from .device import Device
from .llm_client import LLMClient
from .loggingx import init_logger
from .ocr import OCRClient
from .planner import Planner
from .resolver import Resolver
from ..tasks.commissions import CommissionsTask
from ..tasks.currencies import CurrenciesTask
from ..tasks.pickups import PickupsTask


def bootstrap_from_config(config_path: str | Path) -> Dict[str, Any]:
    """Bootstrap all components from configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing all initialized components
    """
    # Load configuration
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = create_default_config()
    
    return bootstrap_from_config_object(config)


def bootstrap_from_config_object(config: AppConfig) -> Dict[str, Any]:
    """Bootstrap all components from configuration object.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary containing all initialized components
    """
    # Initialize logging
    data_dir = config.data_dir
    log_file = data_dir / "logs" / "azlbot.log"
    init_logger(config.logging.level, log_file)
    
    logger.info("Bootstrapping Azur Lane Bot components")
    
    # Initialize device
    device = Device(config.emulator.adb_serial)
    
    # Check device connection
    if not device.is_connected():
        logger.error(f"Device not connected: {config.emulator.adb_serial}")
        if not device.reconnect():
            raise RuntimeError(f"Cannot connect to device: {config.emulator.adb_serial}")
    
    # Initialize components
    capture = Capture(device)
    ocr = OCRClient(config.resolver)
    
    # Initialize LLM client (may fail if API key not set)
    try:
        llm = LLMClient(config.llm)
    except ValueError as e:
        logger.warning(f"LLM client initialization failed: {e}")
        llm = None
    
    # Initialize resolver
    templates_dir = Path(__file__).parent.parent.parent / "config" / "templates"
    resolver = Resolver(config.resolver, ocr, templates_dir)
    
    # Initialize datastore
    db_path = data_dir / "azl.db"
    datastore = DataStore(db_path)
    
    # Initialize actuator
    actuator = Actuator(device, "adb")  # Only ADB backend for now
    
    # Initialize planner
    if llm:
        planner = Planner(device, capture, resolver, ocr, llm, datastore, actuator)
    else:
        planner = None
    
    # Create task instances
    tasks = {
        "currencies": CurrenciesTask(),
        "pickups": PickupsTask(),
        "commissions": CommissionsTask()
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
        "tasks": tasks
    }
    
    logger.info("Bootstrap completed successfully")
    return components


def run_task_cli(task_name: str = None) -> None:
    """CLI entry point for running tasks.
    
    Args:
        task_name: Name of task to run (currencies, pickups, commissions)
    """
    if task_name is None and len(sys.argv) > 1:
        task_name = sys.argv[1]
    
    if not task_name:
        print("Usage: azl-task <task_name>")
        print("Available tasks: currencies, pickups, commissions")
        return
    
    # Find configuration file
    config_paths = [
        Path("config/app.yaml"),
        Path("~/.azlbot/config.yaml").expanduser(),
        Path("/etc/azlbot/config.yaml")
    ]
    
    config_path = None
    for path in config_paths:
        if path.exists():
            config_path = path
            break
    
    if not config_path:
        logger.error("No configuration file found. Please create config/app.yaml")
        return
    
    try:
        # Bootstrap components
        components = bootstrap_from_config(config_path)
        
        planner = components["planner"]
        if not planner:
            logger.error("Planner not available (LLM client failed to initialize)")
            return
        
        tasks = components["tasks"]
        
        # Get task
        if task_name not in tasks:
            logger.error(f"Unknown task: {task_name}")
            logger.info(f"Available tasks: {list(tasks.keys())}")
            return
        
        task = tasks[task_name]
        
        # Run task
        logger.info(f"Running task: {task_name}")
        success = planner.run_task(task)
        
        if success:
            logger.info(f"Task {task_name} completed successfully")
        else:
            logger.error(f"Task {task_name} failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        sys.exit(1)


def test_components(config_path: str | Path = None) -> bool:
    """Test that all components can be initialized.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        True if all components initialized successfully
    """
    try:
        if config_path:
            components = bootstrap_from_config(config_path)
        else:
            config = create_default_config()
            # Override device serial for testing
            config.emulator.adb_serial = "127.0.0.1:5555"
            components = bootstrap_from_config_object(config)
        
        # Test device connection
        device = components["device"]
        if not device.is_connected():
            logger.warning("Device not connected - this is expected if no emulator is running")
        
        # Test capture (will fail without device)
        try:
            capture = components["capture"]
            frame = capture.grab()
            logger.info(f"Captured frame: {frame.full_w}x{frame.full_h}")
        except Exception as e:
            logger.warning(f"Capture test failed (expected without device): {e}")
        
        # Test OCR
        ocr = components["ocr"]
        logger.info(f"OCR backend: {ocr.backend}")
        
        # Test resolver
        resolver = components["resolver"]
        logger.info(f"Resolver loaded {len(resolver._templates)} templates")
        
        # Test datastore
        datastore = components["datastore"]
        test_run_id = datastore.insert_run("test", "test_device")
        logger.info(f"Database test successful, run_id: {test_run_id}")
        
        logger.info("Component test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Component test failed: {e}")
        return False


if __name__ == "__main__":
    run_task_cli()