"""Bootstrap and component initialization.

This module provides bootstrap hooks for initializing all components.
Combines minimal approach with dataset capture and full initialization.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from .actuator import Actuator
from .capture import Capture
from .configs import AppConfig, load_config, create_default_config
from .datastore import DataStore
from .dataset_capture import DatasetCapture
from .device import Device
from .llm_client import LLMClient
from .ocr import OCRClient
from .resolver import Resolver

# Optional imports for full bootstrap
try:
    from .planner import Planner
    HAS_PLANNER = True
except ImportError:
    HAS_PLANNER = False

try:
    from .screens import ScreenStateMachine
    HAS_SCREEN_STATE = True
except (ImportError, AttributeError):
    HAS_SCREEN_STATE = False


def bootstrap_from_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load config from file and wire core components.

    Args:
        config_path: Path to config file, or None to use default

    Returns:
        Dictionary of initialized components
    """
    # Load configuration
    if config_path:
        config = load_config(config_path)
    else:
        config_path_str = os.getenv("AZL_CONFIG", "./config/app.yaml")
        if Path(config_path_str).exists():
            config = load_config(config_path_str)
        else:
            logger.warning(f"Config not found at {config_path_str}, using defaults")
            config = create_default_config()
    
    return bootstrap_from_config_object(config)


def bootstrap_from_config_object(config: AppConfig) -> Dict[str, Any]:
    """Wire core components based on an AppConfig object.
    
    This is the main initialization function that creates all components.
    """
    logger.info("Bootstrapping components from configuration")
    
    # Initialize device
    device = Device(
        adb_serial=config.emulator.adb_serial,
        package_name=config.emulator.package_name
    )
    
    # Initialize dataset capture if enabled
    dataset_capture = None
    if config.data.capture_dataset.enabled:
        dataset_capture = DatasetCapture(
            config=config.data.capture_dataset.model_dump(),
            base_dir=config.data_dir
        )
        logger.info("Dataset capture enabled")
    
    # Initialize capture
    capture = Capture(device, dataset_capture=dataset_capture)
    
    # Initialize OCR
    ocr = OCRClient(config.resolver)
    
    # Initialize LLM
    llm = None
    try:
        llm = LLMClient(config.llm)
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
    datastore = DataStore(config.data_dir / "azl.sqlite3")
    
    # Initialize actuator
    actuator = Actuator(device)
    # Back-link capture for actuator for active_rect usage
    actuator.capture = capture
    
    # Components dictionary
    components: Dict[str, Any] = {
        "config": config,
        "device": device,
        "capture": capture,
        "ocr": ocr,
        "llm": llm,
        "resolver": resolver,
        "datastore": datastore,
        "actuator": actuator,
        "hasher": capture.hasher,
        "dataset_capture": dataset_capture,
    }
    
    # Initialize planner if available
    if HAS_PLANNER and llm is not None:
        try:
            planner = Planner(
                resolver=resolver,
                actuator=actuator,
                datastore=datastore,
                llm=llm,
                ocr=ocr
            )
            components["planner"] = planner
            logger.info("Planner initialized")
        except Exception as e:
            logger.warning(f"Could not initialize planner: {e}")
    
    # Initialize screen state machine if available
    if HAS_SCREEN_STATE:
        try:
            screen_state_machine = ScreenStateMachine()
            components["screen_state_machine"] = screen_state_machine
        except Exception as e:
            logger.warning(f"Could not initialize screen state machine: {e}")
    
    # Initialize tasks if available
    try:
        from azl_bot.tasks import currencies, pickups, commissions
        tasks = {
            "currencies": currencies,
            "pickups": pickups,
            "commissions": commissions
        }
        components["tasks"] = tasks
    except ImportError as e:
        logger.debug(f"Tasks not loaded: {e}")
    
    logger.info(f"Bootstrapped {len(components)} components")
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
