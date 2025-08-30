"""Configuration management for Azur Lane Bot."""

import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field


class EmulatorConfig(BaseModel):
    """Emulator configuration."""
    kind: Literal["waydroid"] = "waydroid"
    adb_serial: str = "127.0.0.1:5555"
    package_name: str = "com.YoStarEN.AzurLane"


class DisplayConfig(BaseModel):
    """Display and capture configuration."""
    target_fps: int = 2
    orientation: Literal["landscape", "portrait"] = "landscape"
    force_resolution: Optional[str] = None


class LLMConfig(BaseModel):
    """LLM client configuration."""
    provider: Literal["gemini"] = "gemini"
    model: str = "flash-2.5"
    endpoint: str = "https://generativelanguage.googleapis.com/v1beta"
    api_key_env: str = "GEMINI_API_KEY"
    max_tokens: int = 2048
    temperature: float = 0.1


class ResolverThresholds(BaseModel):
    """Thresholds for selector resolution."""
    ocr_text: float = 0.75
    ncc_edge: float = 0.60
    ncc_gray: float = 0.70
    orb_inliers: int = 12
    combo_accept: float = 0.65


class ResolverRegions(BaseModel):
    """Predefined regions for selector resolution."""
    top_bar: tuple[float, float, float, float] = (0.00, 0.00, 1.00, 0.12)
    bottom_bar: tuple[float, float, float, float] = (0.00, 0.85, 1.00, 0.15)
    left_panel: tuple[float, float, float, float] = (0.00, 0.12, 0.20, 0.73)
    center: tuple[float, float, float, float] = (0.20, 0.12, 0.60, 0.73)
    right_panel: tuple[float, float, float, float] = (0.80, 0.12, 0.20, 0.73)


class ResolverConfig(BaseModel):
    """Resolver configuration."""
    ocr: Literal["paddle", "tesseract"] = "paddle"
    thresholds: ResolverThresholds = Field(default_factory=ResolverThresholds)
    regions: ResolverRegions = Field(default_factory=ResolverRegions)


class DataConfig(BaseModel):
    """Data storage configuration."""
    base_dir: str = "~/.azlbot"


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    keep_frames: bool = True
    overlay_draw: bool = True


class UIConfig(BaseModel):
    """UI configuration."""
    show_llm_json: bool = False
    zoom_overlay: bool = True


class AppConfig(BaseModel):
    """Main application configuration."""
    emulator: EmulatorConfig = Field(default_factory=EmulatorConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    resolver: ResolverConfig = Field(default_factory=ResolverConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    
    @property
    def data_dir(self) -> Path:
        """Get expanded data directory path."""
        return Path(self.data.base_dir).expanduser()
    
    @property 
    def llm_api_key(self) -> str:
        """Get LLM API key from environment."""
        key = os.getenv(self.llm.api_key_env)
        if not key:
            raise ValueError(f"LLM API key not found in environment variable: {self.llm.api_key_env}")
        return key


def load_config(config_path: str | Path) -> AppConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Parsed configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        pydantic.ValidationError: If config validation fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    return AppConfig(**config_data)


def create_default_config() -> AppConfig:
    """Create default configuration.
    
    Returns:
        Default configuration object
    """
    return AppConfig()


def save_config(config: AppConfig, config_path: str | Path) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        config_path: Path where to save the configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict and save as YAML
    config_dict = config.model_dump()
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)