# AGENTS.md — Best Practices for AI-Assisted Development

This guide provides best practices for AI coding assistants (GitHub Copilot, Claude, GPT-4, etc.) working on the Azur Lane Bot project. Following these guidelines ensures code quality, architectural consistency, and maintainability.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture Principles](#architecture-principles)
- [Coordinate System](#coordinate-system)
- [Code Style & Conventions](#code-style--conventions)
- [Dependency Management](#dependency-management)
- [Testing Guidelines](#testing-guidelines)
- [Common Patterns](#common-patterns)
- [Emulator Constraints](#emulator-constraints)
- [Error Handling](#error-handling)
- [Documentation Standards](#documentation-standards)
- [Performance Considerations](#performance-considerations)
- [Security & Privacy](#security--privacy)

---

## 🎯 Project Overview

### What This Is
An **intelligent, multi-modal automation framework** for the mobile game Azur Lane, combining:
- LLM reasoning (Gemini) for high-level planning
- Dual OCR engines (PaddleOCR/Tesseract) for text extraction
- Template matching for icon detection
- ORB feature matching for robust element location
- LLM vision as final arbiter when methods disagree

### Core Loop Architecture
```
Sense (Capture) → Think (LLM Plan) → Resolve (Multi-Modal Vision) → Act (ADB) → Check (Verify)
```

### Key Design Goals
1. **Resolution-Agnostic**: Works on any screen resolution without reconfiguration
2. **Selector-Based**: Abstract selectors (text/icon/region) instead of pixel coordinates
3. **Confidence-Based**: Multi-method detection with confidence scoring and LLM arbitration
4. **Observable**: Full action logging, frame capture, and UI visualization
5. **Maintainable**: Modular architecture with clear component boundaries

---

## 🏛️ Architecture Principles

### 1. Separation of Concerns

Each component has a single, well-defined responsibility:

| Component | Responsibility | Dependencies |
|-----------|---------------|--------------|
| **Device** | ADB communication | None |
| **Capture** | Frame grabbing, letterbox detection | Device, Hasher |
| **OCR** | Text extraction | None (optional engines) |
| **Resolver** | Element location, confidence scoring | OCR, LLM (optional) |
| **LLM Client** | Plan generation, arbitration | Network (API) |
| **Planner** | Task orchestration, retry logic | Resolver, Actuator, Datastore |
| **Actuator** | Input execution (tap/swipe) | Device, Capture |
| **Datastore** | Persistence | SQLAlchemy |

### 2. Dependency Injection

Components are wired in `bootstrap.py`:
```python
def bootstrap_from_config_object(config: AppConfig) -> Dict[str, Any]:
    device = Device(config.emulator.adb_serial)
    capture = Capture(device)
    ocr = OCRClient(config.resolver)
    llm = LLMClient(config.llm)
    resolver = Resolver(config.resolver.model_dump(), ocr, templates_dir="config/templates", llm=llm)
    actuator = Actuator(device)
    actuator.capture = capture  # Back-link for active_rect
    datastore = DataStore(config.data_dir / "azl.sqlite3")
    planner = Planner(resolver, actuator, datastore, llm, ocr)
    
    return {
        "config": config,
        "device": device,
        "capture": capture,
        "ocr": ocr,
        "llm": llm,
        "resolver": resolver,
        "datastore": datastore,
        "actuator": actuator,
        "planner": planner,
    }
```

**Rule**: Never instantiate dependencies inside components. Always pass them as constructor arguments.

### 3. Configuration Management

All configuration lives in `config/app.yaml` (YAML) → `AppConfig` (Pydantic models).

**Good**:
```python
class ResolverConfig(BaseModel):
    ocr: Literal["paddle", "tesseract"] = "paddle"
    thresholds: ThresholdsConfig
```

**Bad**:
```python
OCR_ENGINE = "paddle"  # Hardcoded constant
```

---

## 📐 Coordinate System

### The Golden Rule
**All internal coordinates are normalized [0.0-1.0] within the active game viewport.**

### Transformation Pipeline

```
Raw Screenshot (e.g., 1920x1080)
    ↓
Letterbox Detection → Active Viewport (e.g., 1920x1000, offset_y=40)
    ↓
Vision Processing → Pixel coords within viewport
    ↓
Normalization → [0.0-1.0] space
    ↓
LLM Planning → Works in normalized space
    ↓
Denormalization → Device pixels for ADB input
```

### Implementation Details

**Normalization** (in `Resolver`):
```python
def normalize_point(self, pixel_x: int, pixel_y: int, viewport_rect) -> Tuple[float, float]:
    """Convert viewport pixel coords to normalized [0.0-1.0]."""
    vp_x, vp_y, vp_w, vp_h = viewport_rect
    norm_x = (pixel_x - vp_x) / vp_w
    norm_y = (pixel_y - vp_y) / vp_h
    return (norm_x, norm_y)
```

**Denormalization** (in `Actuator`):
```python
def tap_norm(self, norm_x: float, norm_y: float, active_rect):
    """Convert normalized coords to device pixels and tap."""
    x, y, w, h = active_rect
    device_x = int(x + norm_x * w)
    device_y = int(y + norm_y * h)
    self.tap(device_x, device_y)
```

### LLM Prompt Requirements

Always include in system prompts:
```
Coordinate System:
- Origin (0.0, 0.0) = top-left of game viewport
- (1.0, 1.0) = bottom-right of game viewport
- (0.5, 0.5) = center of game viewport
- All coordinates must be in range [0.0, 1.0]
```

### Validation

Always validate coordinates:
```python
def validate_point(point: Tuple[float, float]):
    assert 0.0 <= point[0] <= 1.0, f"X coordinate {point[0]} out of bounds"
    assert 0.0 <= point[1] <= 1.0, f"Y coordinate {point[1]} out of bounds"
```

---

## 🎨 Code Style & Conventions

### General Style
- **Formatter**: Black (line length: 88)
- **Import Sort**: isort (profile: black)
- **Linter**: Ruff (extends Black/isort)
- **Type Hints**: Use where practical, especially for public APIs

### Naming Conventions

```python
# Classes: PascalCase
class DeviceInterface:
    pass

# Functions/Methods: snake_case
def normalize_coordinates(x: float, y: float) -> Tuple[float, float]:
    pass

# Constants: UPPER_SNAKE_CASE
DEFAULT_TIMEOUT = 30

# Private members: _leading_underscore
def _internal_helper(self):
    pass

# Type aliases: PascalCase
PointNorm = Tuple[float, float]
```

### Docstrings

Use Google-style docstrings:

```python
def resolve(self, target: Target, frame: Frame) -> Optional[Candidate]:
    """Resolve target location using multiple detection methods.
    
    Args:
        target: Target specification (text/icon/region)
        frame: Current screen frame with active_rect
        
    Returns:
        Best candidate or None if not found
        
    Raises:
        ValueError: If target.kind is invalid
    """
    pass
```

### Imports Organization

```python
# Standard library
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party
import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel

# Local
from .capture import Frame
from .configs import AppConfig
```

### Logging

Use loguru with appropriate levels:

```python
from loguru import logger

# Informational
logger.info(f"Resolved target at ({x:.3f}, {y:.3f}) with confidence {conf:.3f}")

# Warnings
logger.warning("OCR confidence below threshold, using template fallback")

# Errors
logger.error(f"Failed to resolve target: {target}")

# Debug (verbose)
logger.debug(f"Template matching: {len(matches)} candidates found")
```

---

## 📦 Dependency Management

### CRITICAL: Use UV, Not Pip

This project uses **UV** for dependency management. **DO NOT use pip install directly.**

```bash
# ✅ Correct
uv pip install -e .
uv add requests

# ❌ Wrong
pip install -e .
pip install requests
```

### Adding Dependencies

1. **For production dependencies**:
   ```bash
   # Add to pyproject.toml dependencies
   # Then install
   uv pip install -e .
   ```

2. **For optional feature groups**:
   ```python
   # pyproject.toml
   [project.optional-dependencies]
   ui = ["PySide6>=6.6.0"]
   ocr-paddle = ["paddlepaddle>=2.5.0", "paddleocr>=2.7.0"]
   ```

3. **For development tools**:
   ```bash
   uv pip install pytest black ruff mypy
   ```

### Updating Dependencies

```bash
# Update lock file
uv lock --upgrade

# Sync environment
uv pip sync uv.lock
```

---

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── test_device.py         # Unit tests for Device
├── test_capture.py        # Unit tests for Capture
├── test_resolver.py       # Vision tests with fixtures
├── test_config_validation.py  # Config schema validation
└── integration/
    └── test_full_task.py  # End-to-end task execution
```

### Test Fixtures

Use fixtures for test images:

```python
# tests/fixtures/frame_home.png
# tests/fixtures/frame_commissions.png

def test_screen_detection():
    frame = load_fixture_frame("frame_home.png")
    screen = identify_screen(frame)
    assert screen == "home"
```

### Mocking External Dependencies

```python
from unittest.mock import Mock, patch

def test_actuator_tap():
    device_mock = Mock()
    actuator = Actuator(device_mock)
    actuator.tap(500, 500)
    device_mock.input_tap.assert_called_once_with(500, 500)
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_resolver.py::test_text_detection -v

# Skip emulator-dependent tests
pytest tests/ -k "not emulator"

# With coverage
pytest tests/ --cov=azl_bot --cov-report=html
```

---

## 🔧 Common Patterns

### Pattern 1: Multi-Method Resolution with Arbitration

```python
def resolve(self, target: Target, frame: Frame) -> Optional[Candidate]:
    """Resolve using multiple methods, LLM arbitrates on disagreement."""
    candidates = []
    
    # Gather candidates from all methods
    if target.kind == "text":
        candidates.extend(self._detect_by_ocr(target, frame))
    if target.kind == "icon":
        candidates.extend(self._detect_by_template(target, frame))
    
    # ORB features for all types
    candidates.extend(self._detect_by_orb(target, frame))
    
    if not candidates:
        return None
    
    # Sort by confidence
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    
    # If methods disagree significantly, use LLM
    if len(candidates) > 1 and max(c.confidence for c in candidates) < 0.8:
        return self._arbitrate_with_llm(target, frame, candidates)
    
    return candidates[0]
```

### Pattern 2: Frame Stability Detection

```python
def wait_for_stable_screen(self, max_wait: float = 5.0) -> Frame:
    """Wait until screen stops changing (hash stability)."""
    last_hash = None
    stable_count = 0
    required_stable = 2
    
    start = time.time()
    while time.time() - start < max_wait:
        frame = self.capture.grab()
        current_hash = self.hasher.compute_hash(frame.image_bgr)
        
        if last_hash and self.hasher.hamming_distance(last_hash, current_hash) <= 3:
            stable_count += 1
            if stable_count >= required_stable:
                return frame
        else:
            stable_count = 0
        
        last_hash = current_hash
        time.sleep(1.0 / self.idle_fps)
    
    raise TimeoutError("Screen did not stabilize")
```

### Pattern 3: Task Protocol

```python
class Task(Protocol):
    """Protocol for game tasks."""
    name: str
    
    def goal(self) -> Dict[str, Any]:
        """Return goal description for LLM."""
        ...
    
    def success(self, frame: Frame, context: Dict[str, Any]) -> bool:
        """Check if task succeeded."""
        ...
    
    def on_success(self, planner, frame: Frame) -> None:
        """Handle success (e.g., record to database)."""
        ...
```

### Pattern 4: Retry with Exponential Backoff

```python
def execute_with_retry(self, action: Callable, max_retries: int = 3) -> bool:
    """Execute action with exponential backoff."""
    for attempt in range(max_retries):
        try:
            result = action()
            if result:
                return True
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
        
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # 1, 2, 4 seconds
            logger.info(f"Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    return False
```

---

## 🎮 Emulator Constraints

### CRITICAL: ADB-Only Interface

**This project uses STANDARD ADB ONLY.** No emulator-specific APIs.

### Forbidden (Genymotion Free Edition)

```bash
# ❌ These will FAIL with Error 14 on free Genymotion
gmtool admin create
gmtool admin start
gmtool admin stop
gmtool admin list
```

### Allowed Operations

```bash
# ✅ Standard ADB works on ALL emulators
adb connect 192.168.XX.XX:5555
adb devices
adb shell screencap -p /sdcard/screen.png
adb exec-out screencap -p > screenshot.png
adb shell input tap 500 500
adb shell input swipe 100 100 500 500 1000
adb shell input keyevent 4  # BACK
adb shell input keyevent 3  # HOME
```

### Supported Emulators

- **Waydroid** (Linux native Android container)
- **Genymotion Personal/Free** (standard ADB)
- **MEmu** (over network via ADB)
- **Any ADB-compatible device**

### Implementation

```python
class Device:
    """ADB-only device interface - works with all emulators."""
    
    def __init__(self, serial: str):
        self.serial = serial
        subprocess.run(['adb', 'connect', self.serial], check=True)
    
    def screencap_png(self) -> bytes:
        cmd = ['adb', '-s', self.serial, 'exec-out', 'screencap', '-p']
        return subprocess.run(cmd, capture_output=True, check=True).stdout
```

---

## 🛡️ Error Handling

### Hierarchical Error Recovery

```python
try:
    # Primary method
    candidate = self.resolver.resolve(target, frame)
    if candidate and candidate.confidence > 0.7:
        return candidate
except OCRError:
    logger.warning("OCR failed, trying template matching")

try:
    # Fallback method
    candidate = self.resolver._detect_by_template(target, frame)
    if candidate:
        return candidate
except TemplateError:
    logger.warning("Template matching failed, trying LLM vision")

try:
    # LLM arbitration
    return self.llm.locate_element(target, frame)
except LLMError:
    logger.error("All detection methods failed")
    return None
```

### State Recovery

```python
def recover_to_home(self) -> bool:
    """Attempt to navigate back to home screen."""
    # Try back button up to 3 times
    for _ in range(3):
        self.actuator.key_back()
        time.sleep(1.0)
        frame = self.capture.grab()
        if self.identify_screen(frame) == "home":
            return True
    
    # Try home button
    self.actuator.key_home()
    time.sleep(2.0)
    frame = self.capture.grab()
    return self.identify_screen(frame) == "home"
```

### Logging Errors for Debugging

```python
try:
    result = self.resolver.resolve(target, frame)
except Exception as e:
    # Log full context
    logger.error(
        f"Resolution failed | target={target} | frame_shape={frame.image_bgr.shape} | error={e}",
        exc_info=True
    )
    # Save debug frame
    if self.config.logging.keep_frames:
        cv2.imwrite(f"debug_frame_{time.time()}.png", frame.image_bgr)
    raise
```

---

## 📝 Documentation Standards

### Module Docstrings

```python
"""UI element resolution and detection.

This module implements multi-modal element detection combining:
- OCR text matching with fuzzy search
- Template matching (edge + grayscale)
- ORB feature matching
- LLM vision as final arbiter

All methods work in normalized coordinate space [0.0-1.0].
"""
```

### Class Docstrings

```python
class Resolver:
    """Resolves UI element locations using multiple detection methods.
    
    The resolver combines OCR, template matching, and ORB features to locate
    elements on screen. When methods disagree, the LLM client (if available)
    acts as final arbiter using vision capabilities.
    
    All coordinates are returned in normalized [0.0-1.0] space within the
    active viewport (after letterbox removal).
    
    Attributes:
        config: Resolver configuration (thresholds, regions)
        ocr: OCR client instance
        llm: Optional LLM client for arbitration
        templates_dir: Path to template image directory
    """
```

### Method Docstrings

```python
def resolve(self, target: Target, frame: Frame) -> Optional[Candidate]:
    """Resolve target location using multiple detection methods.
    
    Attempts detection in priority order:
    1. OCR text matching (if target.kind == "text")
    2. Template matching (if target.kind == "icon")
    3. ORB feature matching (always)
    4. LLM vision arbitration (if methods disagree)
    
    Args:
        target: Target specification with kind, value, region_hint
        frame: Current screen frame with active_rect and image_bgr
        
    Returns:
        Best candidate with point, confidence, and method,
        or None if not found
        
    Raises:
        ValueError: If target.kind is not supported
        
    Example:
        >>> target = Target(kind="text", value="Commissions", region_hint="bottom_bar")
        >>> candidate = resolver.resolve(target, frame)
        >>> if candidate and candidate.confidence > 0.7:
        >>>     actuator.tap_norm(*candidate.point, frame.active_rect)
    """
```

### Configuration Documentation

Always document config options:

```yaml
resolver:
  thresholds:
    ocr_text: 0.75        # Fuzzy match threshold for OCR text (0.0-1.0)
    ncc_edge: 0.60        # Normalized cross-correlation for edge templates
    ncc_gray: 0.70        # NCC for grayscale templates
    orb_inliers: 12       # Minimum ORB feature inliers required
    combo_accept: 0.65    # Ensemble acceptance threshold
```

---

## ⚡ Performance Considerations

### Frame Rate Management

```python
class PerformanceConfig(BaseModel):
    active_fps: float = 2.0      # Max FPS during interaction
    idle_fps: float = 0.5         # FPS during loading/waiting
    transition_time: float = 5.0  # Max wait for transitions
```

**Guideline**: Never exceed 2 FPS. The game doesn't update that fast, and higher rates waste CPU.

### Hash-Based Change Detection

```python
def should_process_frame(self, frame: Frame) -> bool:
    """Skip processing if frame unchanged (via perceptual hash)."""
    current_hash = self.hasher.compute_hash(frame.image_bgr)
    
    if self.last_hash:
        distance = self.hasher.hamming_distance(self.last_hash, current_hash)
        if distance <= self.hamming_threshold:
            logger.debug("Frame unchanged, skipping processing")
            return False
    
    self.last_hash = current_hash
    return True
```

**Savings**: ~80% of frames during loading screens are identical.

### Caching Vision Results

```python
@lru_cache(maxsize=128)
def get_template(self, template_name: str) -> np.ndarray:
    """Load and cache template images."""
    path = self.templates_dir / f"{template_name}.png"
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
```

### Regional Processing

```python
def ocr_in_region(self, frame: Frame, region: str) -> List[Dict]:
    """Run OCR only in specified region (e.g., 'top_bar')."""
    x, y, w, h = self.regions[region]
    roi = frame.image_bgr[int(y*frame.h):int((y+h)*frame.h), 
                           int(x*frame.w):int((x+w)*frame.w)]
    return self.ocr.extract_text(roi)
```

**Guideline**: Use region hints in targets to limit search space.

---

## 🔒 Security & Privacy

### Prohibited Actions

1. **❌ Never commit secrets** (API keys, passwords) to source code
2. **❌ Never log sensitive data** (user accounts, credentials)
3. **❌ Never share screen captures** containing personal info
4. **❌ Never intercept network traffic** or modify game data

### API Key Handling

```python
# ✅ Correct: Environment variable
api_key = os.getenv("GEMINI_API_KEY")

# ❌ Wrong: Hardcoded
api_key = "AIzaSy..."
```

### Data Collection

Only collect:
- Game state (screen names, button locations)
- In-game resources (Oil, Coins, Gems)
- Action logs (taps, swipes with timestamps)

Never collect:
- User accounts or passwords
- Personal information
- Payment details
- Network traffic

### Local-Only Operation

All vision processing runs **locally**:
- OCR engines (PaddleOCR/Tesseract)
- Template matching (OpenCV)
- ORB features (OpenCV)
- Image hashing

Only **LLM API calls** go over network:
- Plan generation
- Element arbitration

---

## 🚀 Quick Reference

### Project Commands

```bash
# Run GUI
./scripts/run_gui.sh

# Run specific task
./scripts/run_task.sh currencies

# Test components
python -c "from azl_bot.core.bootstrap import test_components; test_components()"

# Run tests
pytest tests/ -v

# Format code
black azl_bot/ tests/
isort azl_bot/ tests/

# Lint
ruff check azl_bot/ tests/

# Type check
mypy azl_bot/core/
```

### Key Files

```
azl_bot/core/bootstrap.py    # Component wiring
azl_bot/core/planner.py      # Task orchestration
azl_bot/core/resolver.py     # Multi-modal detection
azl_bot/core/configs.py      # Configuration schemas
config/app.yaml              # Main configuration
```

### Common Tasks

**Add a new task**:
1. Create `azl_bot/tasks/new_task.py`
2. Implement Task protocol (goal, success, on_success)
3. Register in `bootstrap.py`

**Add a new template**:
1. Crop template from screenshot
2. Save as `config/templates/my_template.png`
3. Use in target: `Target(kind="icon", value="my_template")`

**Add text synonyms**:
1. Edit `config/selectors/synonyms.yaml`
2. Add variations: `commission: ["Commission", "Commissions", "Tasks"]`

**Tune confidence thresholds**:
1. Edit `config/app.yaml` → `resolver.thresholds`
2. Lower values = more permissive, higher values = more strict

---

## 📖 Additional Resources

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Design decisions and rationale
- **[IMPLEMENTATION.md](docs/IMPLEMENTATION.md)** — Implementation details
- **[Tasks Playbook](docs/tasks/README.md)** — First run guide
- **[GUI Documentation](docs/GUI_ENHANCEMENTS.md)** — UI features

---

**Remember**: When in doubt, check existing code patterns in `azl_bot/core/` and follow the same style. Consistency is key to maintainability!
