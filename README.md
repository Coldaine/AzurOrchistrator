# Azur Lane Bot — Automated Game Assistant

An automated assistant for the mobile game Azur Lane, designed to run on Linux with Android emulation.

## Features

- **Resolution-agnostic**: Works across different screen resolutions via normalized coordinates
- **Model-agnostic**: Uses selectors and resolvers instead of hardcoded pixel positions
- **Multi-modal vision**: Combines LLM reasoning, OCR, and classical computer vision
- **Always-on desktop UI**: PySide6-based interface with live preview and controls
2. Navigate to Commissions and read/record current commissions  
3. Read and record currency balances (Oil, Coins, Gems; Cubes optional)

## Architecture

The bot follows a **Sense → Think → Act → Check** loop:

1. **Sense**: Capture screen via ADB, detect letterboxing, normalize coordinates
2. **Think**: LLM (Gemini Flash 2.5) proposes minimal action plan in JSON
3. **Resolve**: Convert LLM selectors to coordinates using OCR + template + feature matching
4. **Act**: Execute taps/swipes via ADB/minitouch
5. **Check**: Verify success and continue or recover

## Requirements

- **Target OS**: Linux (tested on Nobara/Fedora)
- **Android Emulator**: Genymotion (Personal Edition)
  - Note: Uses standard ADB interface only
  - Does NOT use Genymotion-specific scripting features

## Installation

### Quick Start
```bash
# Minimal installation (core functionality only)
pip install -e .

# Full installation with all features
pip install -e .[all]
```

### Optional Features
Install only what you need:

```bash
# UI support (PySide6 interface)
pip install -e .[ui]

# OCR engines
pip install -e .[ocr-paddle]    # PaddleOCR (recommended)
pip install -e .[ocr-tesseract]  # Tesseract alternative

# LLM support for AI planning
pip install -e .[llm]

# Development tools
pip install -e .[dev]
```

### System Dependencies
```bash
# Fedora/Nobara
sudo dnf install -y android-tools tesseract

# Debian/Ubuntu
sudo apt-get install -y android-tools-adb tesseract-ocr
```

### Configuration
```bash
# Copy example config
cp config/app.yaml.example config/app.yaml

# Create .env file for API keys
echo "GEMINI_API_KEY=your_key_here" > .env

# Edit config as needed
nano config/app.yaml
```

See [Configuration Examples](docs/config/EXAMPLES.md) for detailed setup guides for Waydroid, Genymotion, and MEmu.

## Usage

### GUI Mode
```bash
./scripts/run_gui.sh
```

The GUI provides:
- **Task Control Sidebar**: Select and run tasks from the registry
- **Live View**: Real-time frame capture with overlay visualizations
- **Candidate Inspector**: View and inspect resolver detection candidates
- **Keyboard Shortcuts**:
  - `Space`: Start/Stop selected task
  - `O`: Toggle overlay display
  - `S`: Save screenshot
- **Overlay Options**: Toggle OCR boxes, template matches, ORB keypoints, regions, and candidates

See [GUI Enhancements Documentation](docs/GUI_ENHANCEMENTS.md) for detailed usage instructions.

![GUI Overlay Features](docs/images/overlay_showcase.png)

### CLI Mode
```bash
./scripts/run_task.sh currencies
./scripts/run_task.sh pickups  
./scripts/run_task.sh commissions
```

### Tasks playbook

For a concise, step-by-step guide to get your first end-to-end run, see:

- docs/tasks/README.md

## Configuration

Edit `config/app.yaml` to configure:


This bot talks to emulators via standard ADB. To target a MEmu instance running on another PC on your LAN:

- Set `emulator.kind` to `memu` (informational) and point `emulator.adb_serial` to the host:port of the MEmu instance, e.g. `192.168.1.50:21503`.
- Ensure the Windows firewall allows inbound TCP to that port and that the instance binds externally (some setups are localhost-only).
- If the device port isn’t reachable directly, you can instead expose the remote ADB server (5037) and set `ADB_SERVER_HOST`/`ADB_SERVER_PORT` in your environment.

Example snippet for `config/app.yaml`:

  adb_serial: "192.168.1.50:21503"  # adjust to your host and instance port
  package_name: "com.YoStarEN.AzurLane"
```


## License

MIT License - see LICENSE file for details.

## Development

### Setup Development Environment

```bash
# Install with development tools
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Code Quality

```bash
# Run linters
ruff check azl_bot/ tests/
black --check azl_bot/ tests/
isort --check azl_bot/ tests/

# Auto-fix issues
ruff check --fix azl_bot/ tests/
black azl_bot/ tests/
isort azl_bot/ tests/

# Type checking
mypy azl_bot/core/
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
python tests/basic_test.py
python tests/test_config_validation.py

# Skip emulator-dependent tests
pytest tests/ -k "not emulator"
```

### Pre-commit Hooks

The project uses pre-commit hooks for automatic code quality checks:

- Trailing whitespace removal
- YAML/TOML validation
- Ruff linting and formatting
- Black code formatting
- isort import sorting
- MyPy type checking

Run manually: `pre-commit run --all-files`
