#!/usr/bin/env bash
# Waydroid setup and verification script

set -e

echo "Azur Lane Bot - Waydroid Setup Script"
echo "====================================="

# Check if ADB is installed
if ! command -v adb &> /dev/null; then
    echo "ADB not found. Installing android-tools-adb..."
    sudo apt-get update
    sudo apt-get install -y android-tools-adb
fi

# Default Waydroid ADB address
ADB_SERIAL=${1:-"127.0.0.1:5555"}

echo "Checking ADB connection to: $ADB_SERIAL"

# Connect to Waydroid
echo "Connecting to Waydroid..."
adb connect "$ADB_SERIAL" || true

# Wait a moment for connection
sleep 2

# Check if device is connected
if ! adb -s "$ADB_SERIAL" shell echo "test" &> /dev/null; then
    echo "Error: Cannot connect to device $ADB_SERIAL"
    echo "Make sure Waydroid is running and ADB is enabled"
    echo ""
    echo "To start Waydroid:"
    echo "  waydroid show-full-ui"
    echo ""
    echo "To enable ADB in Waydroid:"
    echo "  1. Open Settings in Waydroid"
    echo "  2. Go to System > Developer options"
    echo "  3. Enable USB debugging"
    exit 1
fi

echo "✓ Device connected successfully"

# Get device info
echo ""
echo "Device Information:"
echo "==================="

DEVICE_MODEL=$(adb -s "$ADB_SERIAL" shell getprop ro.product.model | tr -d '\r')
ANDROID_VERSION=$(adb -s "$ADB_SERIAL" shell getprop ro.build.version.release | tr -d '\r')
echo "Model: $DEVICE_MODEL"
echo "Android: $ANDROID_VERSION"

# Get screen size
SCREEN_SIZE=$(adb -s "$ADB_SERIAL" shell wm size | grep "Physical size" | cut -d: -f2 | tr -d ' \r')
SCREEN_DENSITY=$(adb -s "$ADB_SERIAL" shell wm density | grep "Physical density" | cut -d: -f2 | tr -d ' \r')
echo "Screen: $SCREEN_SIZE"
echo "Density: $SCREEN_DENSITY"

# Test screen capture
echo ""
echo "Testing screen capture..."
TEMP_SCREENSHOT="/tmp/azlbot_test_capture.png"
if adb -s "$ADB_SERIAL" exec-out screencap -p > "$TEMP_SCREENSHOT"; then
    if [ -f "$TEMP_SCREENSHOT" ] && [ -s "$TEMP_SCREENSHOT" ]; then
        echo "✓ Screen capture successful"
        
        # Get image info if possible
        if command -v identify &> /dev/null; then
            IMAGE_INFO=$(identify "$TEMP_SCREENSHOT" | cut -d' ' -f3)
            echo "  Captured image: $IMAGE_INFO"
        fi
        
        rm -f "$TEMP_SCREENSHOT"
    else
        echo "✗ Screen capture failed - empty file"
        exit 1
    fi
else
    echo "✗ Screen capture failed"
    exit 1
fi

# Test input
echo ""
echo "Testing input..."
if adb -s "$ADB_SERIAL" shell input tap 100 100; then
    echo "✓ Input test successful"
else
    echo "✗ Input test failed"
    exit 1
fi

echo ""
echo "Setup complete! Device is ready for Azur Lane Bot."
echo ""
echo "Configuration for app.yaml:"
echo "---------------------------"
echo "emulator:"
echo "  adb_serial: \"$ADB_SERIAL\""
echo "  package_name: \"com.YoStarEN.AzurLane\"  # Adjust for your region"
echo ""
echo "Note: Make sure Azur Lane is installed and can be launched in Waydroid"