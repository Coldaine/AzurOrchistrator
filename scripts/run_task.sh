#!/usr/bin/env bash
# Task runner script

set -e

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <task_name>"
    echo "Available tasks: currencies, pickups, commissions"
    exit 1
fi

TASK_NAME="$1"

# Check if config exists
if [ ! -f "config/app.yaml" ]; then
    echo "Configuration file not found. Creating from example..."
    if [ ! -f "config/app.yaml.example" ]; then
        echo "Error: config/app.yaml.example not found"
        exit 1
    fi
    cp config/app.yaml.example config/app.yaml
    echo "Created config/app.yaml - please edit it with your settings"
    exit 1
fi

# Set config path
export AZL_CONFIG=${AZL_CONFIG:-"./config/app.yaml"}

echo "Running task: $TASK_NAME"
echo "Config: $AZL_CONFIG"

# Run the task
python -m azl_bot.core.bootstrap "$TASK_NAME"