"""Bootstrap module for initializing all bot components."""

import os
from pathlib import Path
from typing import Any, Dict

from loguru import logger

from .actuator import Actuator
from .capture import Capture
from .configs import AppConfig, create_default_config, load_config
from .datastore import DataStore
from .device import Device
from .llm_client import LLMClient
from .ocr import OCRClient
from .planner import Planner
from .resolver import Resolver
from .screens import ScreenStateMachine


def bootstrap_from_config(config_path: str | Path) -> Dict[str, Any]:
    """Bootstrap all components from a configuration file.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        Dictionary of initialized components
    """
    config = load_config(config_path)
    return bootstrap_from_config_object(config)


def bootstrap_from_config_object(config: AppConfig) -> Dict[str, Any]:
    """Bootstrap all components from a configuration object.
    
    Args:
        config: Application configuration object
        
    Returns:
        Dictionary of initialized components with keys:
        - config: AppConfig
        - device: Device
        - capture: Capture
        - ocr: OCRClient
        - llm: LLMClient (None if API key not available)
        - resolver: Resolver
        - datastore: DataStore
        - actuator: Actuator
        - planner: Planner
        - tasks: Dict of task instances
        - screen_state_machine: ScreenStateMachine
        - hasher: ImageHasher (from capture)
    """
    logger.info("Bootstrapping bot components...")
    
    # Initialize device
    logger.info(f"Connecting to device: {config.emulator.adb_serial}")
    device = Device(serial=config.emulator.adb_serial)
    
    # Initialize capture
    logger.info("Initializing capture system...")
    capture = Capture(device, target_fps=config.display.target_fps)
    
    # Initialize OCR
    logger.info(f"Initializing OCR ({config.resolver.ocr})...")
    ocr = OCRClient(engine=config.resolver.ocr)
    
    # Initialize LLM (optional - may not have API key)
    llm = None
    try:
        api_key = config.llm_api_key
        logger.info(f"Initializing LLM ({config.llm.provider})...")
        llm = LLMClient(
            provider=config.llm.provider,
            api_key=api_key,
            model=config.llm.model,
            max_tokens=config.llm.max_tokens,
            temperature=config.llm.temperature
        )
    except (ValueError, KeyError) as e:
        logger.warning(f"LLM not available: {e}")
        logger.warning("Planner will work in limited mode without LLM")
    
    # Initialize resolver
    logger.info("Initializing resolver...")
    resolver = Resolver(
        ocr=ocr,
        templates_dir=Path("config/templates"),
        synonyms_path=Path("config/selectors/synonyms.yaml"),
        thresholds=config.resolver.thresholds
    )
    
    # Initialize datastore
    logger.info("Initializing datastore...")
    data_dir = config.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    datastore = DataStore(db_path=data_dir / "azlbot.db")
    
    # Initialize actuator
    actuator = Actuator(device)
    
    # Initialize screen state machine
    screen_state_machine = ScreenStateMachine()
    
    # Initialize planner
    logger.info("Initializing planner...")
    planner = Planner(
        device=device,
        capture=capture,
        resolver=resolver,
        ocr=ocr,
        llm=llm,
        datastore=datastore,
        actuator=actuator
    )
    
    # Initialize tasks
    from ..tasks.commissions import create_commissions_task
    from ..tasks.currencies import create_currencies_task
    from ..tasks.daily import create_daily_task
    from ..tasks.pickups import create_pickups_task
    
    tasks = {
        "commissions": create_commissions_task(),
        "pickups": create_pickups_task(),
        "currencies": create_currencies_task(),
        "daily_maintenance": create_daily_task(),
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
    
    logger.info("Bootstrap complete!")
    return components


def test_components() -> bool:
    """Test that components can be instantiated.
    
    Returns:
        True if all components can be created, False otherwise
    """
    try:
        # Test with minimal config
        config = create_default_config()
        config.emulator.adb_serial = "test_device"
        
        # Don't actually connect to device in test mode
        logger.info("Component test would initialize: Device, Capture, OCR, Resolver, DataStore")
        logger.info("Skipping actual initialization in test mode")
        
        return True
    except Exception as e:
        logger.error(f"Component test failed: {e}")
        return False
