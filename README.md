# Azur Lane Bot — Automated Game Assistant

An automated assistant for the mobile game Azur Lane, designed to run on Debian Linux with Waydroid emulation.

## Features

- **Resolution-agnostic**: Works across different screen resolutions via normalized coordinates
- **Model-agnostic**: Uses selectors and resolvers instead of hardcoded pixel positions
- **Multi-modal vision**: Combines LLM reasoning, OCR, and classical computer vision
- **Always-on desktop UI**: PySide6-based interface with live preview and controls

## Primary Goals (v0)

1. Collect all easy main-menu pickups
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

- **Target OS**: Debian Linux
- **Android Environment**: Waydroid
- **Python**: ≥ 3.10
- **Dependencies**: See pyproject.toml

## Installation

```bash
# Install system dependencies
sudo apt-get install -y android-tools-adb tesseract-ocr

# Install Python package
pip install -e .

# Configure
cp config/app.yaml.example config/app.yaml
# Edit config/app.yaml with your settings
```

## Usage

### GUI Mode
```bash
./scripts/run_gui.sh
```

### CLI Mode
```bash
./scripts/run_task.sh currencies
./scripts/run_task.sh pickups  
./scripts/run_task.sh commissions
```

## Configuration

Edit `config/app.yaml` to configure:
- ADB device serial and package name
- LLM endpoint and API key
- OCR and vision thresholds
- UI preferences

## License

MIT License - see LICENSE file for details.