# Configuration Examples

This document provides configuration examples and troubleshooting tips for different emulator setups.

## Table of Contents
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Configuration Examples](#configuration-examples)
  - [Waydroid](#waydroid-configuration)
  - [Genymotion](#genymotion-configuration)
  - [MEmu](#memu-configuration)
- [Troubleshooting](#troubleshooting)

## Installation

### Basic Installation
For minimal installation without heavy OCR/UI dependencies:
```bash
pip install -e .
```

### Installation with Extras
Install specific feature sets as needed:

```bash
# UI support (PySide6)
pip install -e .[ui]

# OCR with PaddleOCR (recommended)
pip install -e .[ocr-paddle]

# OCR with Tesseract (alternative)
pip install -e .[ocr-tesseract]

# LLM support for AI planning
pip install -e .[llm]

# Development tools (linting, testing, type checking)
pip install -e .[dev]

# Install everything
pip install -e .[all]
```

### Common Combinations
```bash
# Typical user setup with UI and PaddleOCR
pip install -e .[ui,ocr-paddle,llm]

# Developer setup
pip install -e .[all]
```

## Environment Setup

### Using .env File
Create a `.env` file in the project root to store sensitive configuration:

```bash
# .env
GEMINI_API_KEY=your_api_key_here
```

The configuration system will automatically load variables from `.env` if `python-dotenv` is installed.

### Using Environment Variables
Alternatively, export variables in your shell:

```bash
export GEMINI_API_KEY=your_api_key_here
```

## Configuration Examples

### Waydroid Configuration

Waydroid runs Android in a container on Linux systems.

**config/app.yaml:**
```yaml
emulator:
  kind: waydroid
  adb_serial: "127.0.0.1:5555"  # Default Waydroid ADB
  package_name: "com.YoStarEN.AzurLane"

display:
  target_fps: 2
  orientation: "landscape"
  force_resolution: null  # Let Waydroid use native resolution

llm:
  provider: "gemini"
  model: "gemini-1.5-flash-latest"
  endpoint: "https://generativelanguage.googleapis.com/v1beta"
  api_key_env: "GEMINI_API_KEY"
  max_tokens: 2048
  temperature: 0.1

resolver:
  ocr: "paddle"  # PaddleOCR recommended for better accuracy
  thresholds:
    ocr_text: 0.75
    ncc_edge: 0.60
    ncc_gray: 0.70
    orb_inliers: 12
    combo_accept: 0.65

data:
  base_dir: "~/.azlbot"

logging:
  level: "INFO"
  keep_frames: true
  overlay_draw: true
```

**Setup Steps:**
1. Install Waydroid: https://waydro.id/
2. Initialize: `waydroid init`
3. Start: `waydroid show-full-ui`
4. Install Azur Lane APK
5. Connect ADB: `adb connect 127.0.0.1:5555`
6. Verify: `adb devices`

### Genymotion Configuration

Genymotion Personal Edition (free) or commercial.

**config/app.yaml:**
```yaml
emulator:
  kind: "generic"  # or "genymotion"
  adb_serial: "192.168.56.101:5555"  # Typical Genymotion IP
  package_name: "com.YoStarEN.AzurLane"

display:
  target_fps: 2
  orientation: "landscape"
  force_resolution: "1920x1080"  # Match your device resolution

# ... rest same as Waydroid example
```

**Setup Steps:**
1. Install Genymotion: https://www.genymotion.com/download/
2. Create device through GUI (Personal Edition restriction)
3. Start device through GUI
4. Get device IP from Genymotion UI
5. Connect ADB: `adb connect <device_ip>:5555`
6. Install APK: `adb install AzurLane.apk`

**Important Notes:**
- Personal Edition requires manual device creation via GUI
- Cannot use `gmtool` commands (premium feature)
- All automation must use standard ADB interface

### MEmu Configuration

MEmu is a Windows/Mac Android emulator.

**config/app.yaml:**
```yaml
emulator:
  kind: "memu"
  adb_serial: "127.0.0.1:21503"  # Default MEmu ADB port
  package_name: "com.YoStarEN.AzurLane"

display:
  target_fps: 2
  orientation: "landscape"
  force_resolution: "1280x720"  # Common MEmu resolution

# ... rest same as above
```

**Remote MEmu (LAN):**
```yaml
emulator:
  kind: "memu"
  adb_serial: "192.168.1.50:21503"  # Remote MEmu IP
  package_name: "com.YoStarEN.AzurLane"
```

**Setup Steps:**
1. Install MEmu
2. Enable ADB in MEmu settings
3. Connect: `adb connect 127.0.0.1:21503`
4. Verify: `adb devices`

**Multiple MEmu Instances:**
Each MEmu instance uses a different port (21503, 21513, 21523, etc.)

## Troubleshooting

### Common Issues

#### 1. Config Validation Errors

**Problem:** Configuration file fails validation
```
Configuration validation failed for config/app.yaml:
  • Field 'display -> target_fps': Input should be greater than or equal to 1
    Got value: 0
```

**Solution:** Check the error message for the exact field and expected value. Common issues:
- `target_fps` must be between 1 and 60
- `force_resolution` must be in format "WIDTHxHEIGHT" (e.g., "1920x1080")
- Threshold values must be between 0.0 and 1.0
- Region coordinates must be normalized (0.0 to 1.0)

#### 2. ADB Connection Failed

**Problem:** Cannot connect to emulator
```
error: failed to connect to 127.0.0.1:5555
```

**Solutions:**
- Verify emulator is running: `adb devices`
- Kill and restart ADB server: `adb kill-server && adb start-server`
- Check correct port for your emulator
- For Waydroid: ensure container is running
- For Genymotion: check firewall isn't blocking connection

#### 3. OCR Not Working

**Problem:** Text detection fails

**For PaddleOCR:**
```bash
# Install with CPU support
pip install paddlepaddle
pip install paddleocr

# On first run, models will download (may take time)
```

**For Tesseract:**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Install Python wrapper
pip install pytesseract
```

Set in config:
```yaml
resolver:
  ocr: "paddle"  # or "tesseract"
```

#### 4. LLM API Key Not Found

**Problem:**
```
ValueError: LLM API key not found in environment variable: GEMINI_API_KEY
```

**Solutions:**
1. Create `.env` file in project root:
   ```
   GEMINI_API_KEY=your_key_here
   ```

2. Or export in shell:
   ```bash
   export GEMINI_API_KEY=your_key_here
   ```

3. Verify dotenv is installed:
   ```bash
   pip install python-dotenv
   ```

#### 5. Screen Capture Issues

**Problem:** Black screen or incorrect resolution

**Solutions:**
- Check `force_resolution` matches emulator display
- Try without `force_resolution` (set to `null`)
- Verify ADB screen capture works: `adb exec-out screencap -p > test.png`
- Check emulator graphics mode (use Software/OpenGL)

#### 6. Letterbox Detection Problems

**Problem:** Active area detection crops too much

**Solution:** Adjust in code or disable letterbox detection if emulator doesn't use letterboxing. Check logs for detection warnings.

### Performance Tuning

**Low FPS:**
```yaml
display:
  target_fps: 1  # Reduce if system is slow
```

**Reduce OCR Load:**
```yaml
resolver:
  ocr: "tesseract"  # Lighter than PaddleOCR
```

**Adjust Thresholds:**
If detection is too sensitive/loose:
```yaml
resolver:
  thresholds:
    ocr_text: 0.80  # Increase for stricter matching
    combo_accept: 0.70  # Increase for higher confidence
```

### Getting Help

1. **Check Logs:** Look in `~/.azlbot/` for detailed logs
2. **Validate Config:** Run `python -m azl_bot.core.configs` to test config loading
3. **Test Components:** Run `python tests/basic_test.py`
4. **GitHub Issues:** https://github.com/Coldaine/AzurOrchistrator/issues

### Debug Mode

Enable detailed logging:
```yaml
logging:
  level: "DEBUG"
  keep_frames: true
  overlay_draw: true
```

This will save frames and show detailed debug information.
