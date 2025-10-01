"""Bootstrap and component initialization.

This module provides bootstrap hooks for initializing all components.
Combines dataset capture support with task registry integration.
"""

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
from .resolver import Resolver

# Optional imports
try:
    from .dataset_capture import DatasetCapture
    HAS_DATASET = True
except ImportError:
    HAS_DATASET = False

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
        - planner: Planner (if available)
        - tasks: Dict of task instances
        - screen_state_machine: ScreenStateMachine (if available)
        - hasher: ImageHasher (from capture)
        - dataset_capture: DatasetCapture (if enabled)
    """
    logger.info("Bootstrapping bot components...")
    
    # Initialize device
    logger.info(f"Connecting to device: {config.emulator.adb_serial}")
    device = Device(serial=config.emulator.adb_serial)
    
    # Initialize dataset capture if enabled
    dataset_capture = None
    if HAS_DATASET and config.data.capture_dataset.enabled:
        dataset_capture = DatasetCapture(  # type: ignore[possibly-unbound]
            config=config.data.capture_dataset.model_dump(),
            base_dir=config.data_dir
        )
        logger.info("Dataset capture enabled")
    
    # Initialize capture
    logger.info("Initializing capture system...")
    capture = Capture(device, dataset_capture=dataset_capture)
    
    # Initialize OCR
    logger.info(f"Initializing OCR ({config.resolver.ocr})...")
    ocr = OCRClient(config.resolver)
    
    # Initialize LLM (optional - may not have API key)
    llm = None
    try:
        logger.info(f"Initializing LLM ({config.llm.provider})...")
        llm = LLMClient(config.llm)
    except ValueError as e:
        logger.warning(f"LLM not available: {e}")
        logger.warning("Planner will work in limited mode without LLM")
    
    # Initialize resolver
    logger.info("Initializing resolver...")
    templates_dir = str(Path("./config/templates").absolute())
    resolver = Resolver(
        config=config.resolver.model_dump(),
        ocr_client=ocr,
        templates_dir=templates_dir,
        llm=llm
    )
    
    # Initialize datastore
    logger.info("Initializing datastore...")
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
    if HAS_PLANNER:
        try:
            logger.info("Initializing planner...")
            planner = Planner(  # type: ignore[possibly-unbound]
                device=device,
                capture=capture,
                resolver=resolver,
                ocr=ocr,
                llm=llm if llm else None,  # type: ignore[arg-type]
                datastore=datastore,
                actuator=actuator
            )
            components["planner"] = planner
        except Exception as e:
            logger.warning(f"Could not initialize planner: {e}")
    
    # Initialize screen state machine if available
    if HAS_SCREEN_STATE:
        try:
            screen_state_machine = ScreenStateMachine()  # type: ignore[possibly-unbound]
            components["screen_state_machine"] = screen_state_machine
        except Exception as e:
            logger.warning(f"Could not initialize screen state machine: {e}")
    
    # Initialize tasks using task registry
    try:
        from ..tasks.registry import get_all_tasks  # type: ignore[attr-defined]
        tasks = get_all_tasks()
        components["tasks"] = tasks
        logger.info(f"Loaded {len(tasks)} tasks from registry")
    except ImportError:
        # Fallback to manual task loading
        try:
            from ..tasks import currencies, pickups, commissions
            tasks = {
                "currencies": currencies,
                "pickups": pickups,
                "commissions": commissions
            }
            # Try to add daily task if available
            try:
                from ..tasks import daily
                tasks["daily_maintenance"] = daily
            except ImportError:
                pass
            components["tasks"] = tasks
            logger.info(f"Loaded {len(tasks)} tasks manually")
        except ImportError as e:
            logger.debug(f"Tasks not loaded: {e}")
    
    logger.info(f"Bootstrapped {len(components)} components")
    logger.info("Bootstrap complete!")
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
