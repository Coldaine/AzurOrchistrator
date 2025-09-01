#!/bin/bash
# Launch Genymotion with proper Qt plugin paths

echo "Starting Genymotion with Qt plugin paths configured..."

# Set Qt plugin path to find Wayland plugins
export QT_PLUGIN_PATH=/usr/lib64/qt5/plugins:/usr/lib64/qt6/plugins:$QT_PLUGIN_PATH

# Try to identify which Qt version Genymotion uses
echo "Attempting Wayland first..."
if ! genymotion "$@" 2>/tmp/genymotion_error.log; then
    echo "Wayland failed, trying X11 compatibility mode..."
    echo "Error output:"
    cat /tmp/genymotion_error.log
    echo ""
    echo "Launching with X11 compatibility (XWayland)..."
    QT_QPA_PLATFORM=xcb genymotion "$@"
fi