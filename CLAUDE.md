# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Azur Lane Bot - An automated assistant for the mobile game Azur Lane, designed to run on Linux with Android emulation. Uses a multi-modal vision approach combining LLM reasoning, OCR, and computer vision.

## Architecture
The bot follows a **Sense → Think → Act → Check** loop:
1. **Sense**: Capture screen via ADB, detect letterboxing, normalize coordinates
2. **Think**: LLM (Gemini Flash 2.5) proposes minimal action plan in JSON
3. **Resolve**: Convert LLM selectors to coordinates using OCR + template + feature matching
4. **Act**: Execute taps/swipes via ADB/minitouch
5. **Check**: Verify success and continue or recover

## Key Components
- **Core**: Device control (`device.py`), screen capture (`capture.py`), action execution (`actuator.py`)
- **Vision**: OCR (`ocr.py`), template matching and resolution (`resolver.py`)
- **Planning**: LLM-based action planning (`planner.py`, `llm_client.py`)
- **Tasks**: Automated game tasks (`commissions.py`, `currencies.py`, `pickups.py`)
- **UI**: PySide6-based GUI with live preview (`ui/app.py`)
- **Data**: SQLAlchemy-based persistence (`datastore.py`)

## Development Commands

### Dependency Management - IMPORTANT
**This project uses UV for dependency management. Do NOT use pip directly.**

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create/activate virtual environment
uv venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install project dependencies
uv pip install -e .

# Install development dependencies
uv pip install pytest pytest-asyncio black isort flake8 mypy ipython

# Add new dependencies
uv add <package-name>

# Update dependencies
uv lock --upgrade
```

### Running the Application
```bash
# Ensure UV environment is activated first!
source .venv/bin/activate

# GUI Mode
./scripts/run_gui.sh

# CLI Mode - specific tasks
./scripts/run_task.sh currencies
./scripts/run_task.sh pickups
./scripts/run_task.sh commissions

# Or using Python directly
python -m azl_bot.ui.app                    # GUI
python -m azl_bot.core.bootstrap <task>     # CLI task
```

### Testing
```bash
# Run basic tests
python tests/basic_test.py

# Test component initialization
python -c "from azl_bot.core.bootstrap import test_components; test_components()"
```

### Code Quality
```bash
# Format code with black
black azl_bot/ --line-length 88

# Sort imports
isort azl_bot/ --profile black --line-length 88

# Type checking
mypy azl_bot/ --python-version 3.10

# Linting
flake8 azl_bot/
```

## Configuration
Primary configuration file: `config/app.yaml`
- Copy from `config/app.yaml.example` if not exists
- Set environment variable `AZL_CONFIG` to override config path
- Required: Set `GEMINI_API_KEY` environment variable for LLM functionality

## Emulator Target - CRITICAL RESTRICTIONS
**Using Genymotion Personal Edition (FREE) on Linux**

### FORBIDDEN (Premium Only)
- ❌ **NO gmtool admin commands** (create, start, stop, list)
- ❌ **NO Genymotion Python API**
- ❌ **NO command-line device management**

### ALLOWED (Free Version)
- ✅ **Standard ADB interface only**
- ✅ Device creation/management through GUI only

### Valid Commands
```bash
# Good - Standard ADB
adb connect 192.168.XX.XX:5555
adb shell screencap -p /sdcard/screen.png
adb shell input tap 500 500
adb shell input swipe 100 100 500 500 1000

# Bad - GMTool (will fail with Error 14)
gmtool admin create   # FORBIDDEN
gmtool admin start    # FORBIDDEN
```

## Device Setup
1. Create device manually in Genymotion GUI
2. Start device through GUI
3. Connect via ADB: `adb connect <device_ip>:5555`
4. Run bot automation using standard ADB commands

## Key Configuration Settings
- **ADB Serial**: Default `127.0.0.1:5555` (Waydroid) or Genymotion IP
- **Package Name**: `com.YoStarEN.AzurLane`
- **OCR Engine**: PaddleOCR (default) or Tesseract
- **LLM**: Gemini 1.5 Flash via Google API
- **Data Directory**: `~/.azlbot/`

## Project Structure Conventions
- Resolution-agnostic design using normalized coordinates (0.0-1.0)
- Selector-based element location instead of hardcoded pixels
- Multi-modal vision combining LLM, OCR, and template matching
- Task-based automation structure in `azl_bot/tasks/`
- PySide6 for GUI components
- SQLAlchemy for data persistence
- Loguru for structured logging

## Dependencies
**Managed via UV - see pyproject.toml and uv.lock files**
- Core: `pydantic`, `numpy`, `opencv-python`, `Pillow`, `PySide6`
- OCR: `paddlepaddle`, `paddleocr`, `pytesseract`
- Vision: `rapidfuzz` for text matching
- Development: `pytest`, `black`, `isort`, `flake8`, `mypy`

**IMPORTANT**: Always use UV commands for dependency management:
- `uv add <package>` to add new dependencies
- `uv pip install -e .` to install the project
- Never use `pip install` directly