#!/usr/bin/env bash
# GUI launcher script

set -e

# Check if config exists
if [ ! -f "config/app.yaml" ]; then
    echo "Configuration file not found. Creating from example..."
    if [ ! -f "config/app.yaml.example" ]; then
        echo "Error: config/app.yaml.example not found"
        exit 1
    fi
    cp config/app.yaml.example config/app.yaml
    echo "Created config/app.yaml - please edit it with your settings"
fi

# Set config path
export AZL_CONFIG=${AZL_CONFIG:-"./config/app.yaml"}

echo "Starting Azur Lane Bot GUI..."
echo "Config: $AZL_CONFIG"

# Run the GUI
python -m azl_bot.ui.app