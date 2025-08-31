# Project-Specific Instructions for Claude

## Emulator Target
- **Primary Target**: Genymotion Personal Edition (FREE) on Linux
- **Interface**: Standard ADB commands only

## CRITICAL RESTRICTION - Premium Features FORBIDDEN
**We are using the FREE Personal Edition. The following are FORBIDDEN:**
- ❌ **NO gmtool admin commands** (create, start, stop, list - all require paid license)
- ❌ **NO Genymotion Python API** (premium only)
- ❌ **NO command-line device management** (premium only)
- ✅ **ONLY use standard ADB interface** (adb shell, adb exec-out, etc.)
- ✅ **Device creation/management through GUI only**

## Why This Restriction
GMTool commands require a paid license ($200+/year). Attempting to use them results in Error Code 14.
We must use the GUI for device management and ADB for all automation.

## Allowed Commands
```bash
# Good - Standard ADB (works with free version)
adb connect 192.168.XX.XX:5555
adb shell screencap -p /sdcard/screen.png
adb shell input tap 500 500
adb shell input swipe 100 100 500 500 1000

# Bad - GMTool commands (PREMIUM ONLY - will fail)
gmtool admin create  # ERROR 14: Requires paid license
gmtool admin start   # ERROR 14: Requires paid license
gmtool admin list    # ERROR 14: Requires paid license
```

## Device Setup Process
1. Create device manually in Genymotion GUI
2. Start device through GUI
3. Use ADB for all automation tasks