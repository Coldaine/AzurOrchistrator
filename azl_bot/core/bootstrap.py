"""Bootstrap and component initialization.

This module provides minimal bootstrap hooks needed by the UI and tests.
"""

from pathlib import Path
from typing import Any, Dict

from loguru import logger

from .configs import AppConfig, create_default_config, load_config
from .device import Device
from .capture import Capture
from .ocr import OCRClient
from .llm_client import LLMClient
from .resolver import Resolver
from .datastore import DataStore
from .actuator import Actuator


def bootstrap_from_config(config_path: str | Path) -> Dict[str, Any]:
    """Load config from file and wire core components.

    Returns a components dict consumed by the UI.
    """
    config = load_config(config_path)
    return bootstrap_from_config_object(config)


def bootstrap_from_config_object(config: AppConfig) -> Dict[str, Any]:
    """Wire core components based on an AppConfig object."""
    # Device
    device = Device(config.emulator.adb_serial)
    # Capture
    capture = Capture(device)
    # OCR
    ocr = OCRClient(config.resolver)
    # LLM
    llm = LLMClient(config.llm)
    # Resolver
    resolver = Resolver(config.resolver.model_dump(), ocr, templates_dir=str(Path("config/templates")), llm=llm)
    # Data store
    datastore = DataStore(config.data_dir / "azl.sqlite3")
    # Actuator
    actuator = Actuator(device)
    # Back-link capture for actuator for active_rect usage
    actuator.capture = capture

    components = {
        "config": config,
        "device": device,
        "capture": capture,
        "ocr": ocr,
        "llm": llm,
        "resolver": resolver,
        "datastore": datastore,
        "actuator": actuator,
        # Placeholders for planner/tasks until implemented
        "planner": None,
        "tasks": {},
        "hasher": capture.hasher,
    }

    logger.info("Bootstrap completed")
    return components


def test_components() -> bool:
    """Basic smoke test used by tests/basic_test.py."""
    try:
        config = create_default_config()
        assert config is not None
        return True
    except Exception as e:
        logger.error(f"Component initialization failed: {e}")
        return False
