"""Configuration management for Azur Lane Bot."""

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class EmulatorConfig(BaseModel):
    """Emulator configuration."""
    # Supported kinds are informational; current implementation uses plain ADB.
    # Keeping this flexible lets you label configs (e.g., "waydroid", "memu", "generic").
    kind: Literal["waydroid", "memu", "generic"] = "waydroid"
    adb_serial: str = "127.0.0.1:5555"
    package_name: str = "com.YoStarEN.AzurLane"


class DisplayConfig(BaseModel):
    """Display and capture configuration."""
    model_config = {"validate_assignment": True}
    
    target_fps: int = Field(default=2, ge=1, le=60, description="Target frames per second for capture")
    orientation: Literal["landscape", "portrait"] = "landscape"
    force_resolution: Optional[str] = Field(default=None, description="Force specific resolution (e.g., '1920x1080')")
    
    @field_validator('force_resolution')
    @classmethod
    def validate_resolution(cls, v: Optional[str]) -> Optional[str]:
        """Validate resolution format if provided."""
        if v is not None and v != "":
            if not v or 'x' not in v:
                raise ValueError("Resolution must be in format WxH (e.g., '1920x1080')")
            try:
                w, h = v.split('x')
                width, height = int(w), int(h)
                if width < 100 or height < 100:
                    raise ValueError("Resolution dimensions must be at least 100x100")
                if width > 10000 or height > 10000:
                    raise ValueError("Resolution dimensions must be less than 10000x10000")
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError("Resolution must contain valid integers")
                raise
        return v


class LLMConfig(BaseModel):
    """LLM client configuration."""
    provider: Literal["gemini"] = "gemini"
    model: str = "flash-2.5"
    endpoint: str = "https://generativelanguage.googleapis.com/v1beta"
    api_key_env: str = "GEMINI_API_KEY"
    max_tokens: int = Field(default=2048, ge=100, le=100000, description="Maximum tokens for LLM response")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM temperature for response generation")


class ResolverThresholds(BaseModel):
    """Thresholds for selector resolution."""
    ocr_text: float = Field(default=0.75, ge=0.0, le=1.0, description="OCR confidence threshold")
    ncc_edge: float = Field(default=0.60, ge=0.0, le=1.0, description="Edge-based template matching threshold")
    ncc_gray: float = Field(default=0.70, ge=0.0, le=1.0, description="Grayscale template matching threshold")
    orb_inliers: int = Field(default=12, ge=1, le=1000, description="Minimum ORB feature inliers")
    combo_accept: float = Field(default=0.65, ge=0.0, le=1.0, description="Combined method acceptance threshold")
    # Weights for combining method confidences
    weights: Dict[str, float] = Field(default_factory=lambda: {
        "ocr": 1.0,
        "template": 1.0,
        "orb": 0.9,
        "region_hint": 0.5,
        "llm_arbitration": 1.2
    })


class ResolverRegions(BaseModel):
    """Predefined regions for selector resolution."""
    top_bar: List[float] = Field(default=[0.00, 0.00, 1.00, 0.12], min_length=4, max_length=4, description="Top bar region [x, y, w, h]")
    bottom_bar: List[float] = Field(default=[0.00, 0.85, 1.00, 0.15], min_length=4, max_length=4, description="Bottom bar region [x, y, w, h]")
    left_panel: List[float] = Field(default=[0.00, 0.12, 0.20, 0.73], min_length=4, max_length=4, description="Left panel region [x, y, w, h]")
    center: List[float] = Field(default=[0.20, 0.12, 0.60, 0.73], min_length=4, max_length=4, description="Center region [x, y, w, h]")
    right_panel: List[float] = Field(default=[0.80, 0.12, 0.20, 0.73], min_length=4, max_length=4, description="Right panel region [x, y, w, h]")
    
    @field_validator('top_bar', 'bottom_bar', 'left_panel', 'center', 'right_panel')
    @classmethod
    def validate_region(cls, v: List[float]) -> List[float]:
        """Validate region coordinates are normalized."""
        if len(v) != 4:
            raise ValueError("Region must have exactly 4 values [x, y, w, h]")
        for i, val in enumerate(v):
            if not 0.0 <= val <= 1.0:
                coord_name = ['x', 'y', 'w', 'h'][i]
                raise ValueError(f"Region {coord_name} must be between 0.0 and 1.0, got {val}")
        return v


class ResolverConfig(BaseModel):
    """Resolver configuration."""
    ocr: Literal["paddle", "tesseract"] = "paddle"
    thresholds: ResolverThresholds = Field(default_factory=ResolverThresholds)
    regions: ResolverRegions = Field(default_factory=ResolverRegions)


class DataCaptureConfig(BaseModel):
    """Dataset capture configuration."""
    enabled: bool = False
    sample_rate_hz: float = 0.5
    max_dim: int = 1280
    format: Literal["jpg", "png"] = "jpg"
    jpeg_quality: int = 85
    dedupe: Dict[str, Any] = Field(default_factory=lambda: {
        "method": "dhash",
        "hamming_threshold": 3
    })
    retention: Dict[str, int] = Field(default_factory=lambda: {
        "max_files": 2000,
        "max_days": 60
    })
    metadata: bool = True


class DataConfig(BaseModel):
    """Data storage configuration."""
    base_dir: str = "~/.azlbot"
    capture_dataset: DataCaptureConfig = Field(default_factory=DataCaptureConfig)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    keep_frames: bool = True
    overlay_draw: bool = True


class UIConfig(BaseModel):
    """UI configuration."""
    show_llm_json: bool = False
    zoom_overlay: bool = True


class LoopConfig(BaseModel):
    """Loop execution configuration."""
    target_fps: float = 2.0
    stability_frames: int = 3
    stability_timeout_sec: float = 10.0
    max_retries: int = 3
    retry_backoff_base: float = 1.5
    recovery_enabled: bool = True
    hamming_threshold: float = 0.05


class AppConfig(BaseModel):
    """Main application configuration."""
    emulator: EmulatorConfig = Field(default_factory=EmulatorConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    resolver: ResolverConfig = Field(default_factory=ResolverConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    
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
        ValueError: If config validation fails with detailed field-level errors
    """
    # Load .env file if present
    _load_dotenv()
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please create it from the example: cp {config_path.parent}/app.yaml.example {config_path}"
        )
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML configuration file {config_path}: {e}")
    
    if config_data is None:
        raise ValueError(f"Configuration file {config_path} is empty")
    
    try:
        return AppConfig(**config_data)
    except Exception as e:
        # Provide better error messages for validation errors
        error_msg = f"Configuration validation failed for {config_path}:\n"
        
        # Check if it's a Pydantic validation error
        if hasattr(e, 'errors'):
            for err in e.errors():  # type: ignore[attr-defined]
                field_path = ' -> '.join(str(loc) for loc in err['loc'])
                error_msg += f"  • Field '{field_path}': {err['msg']}\n"
                if 'input' in err:
                    error_msg += f"    Got value: {err['input']}\n"
        else:
            error_msg += f"  {str(e)}\n"
        
        raise ValueError(error_msg) from e


def _load_dotenv() -> None:
    """Load environment variables from .env file if present."""
    try:
        from dotenv import load_dotenv
        # Look for .env in current directory and parent directories
        load_dotenv(verbose=False)
    except ImportError:
        # python-dotenv not installed, skip
        pass


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