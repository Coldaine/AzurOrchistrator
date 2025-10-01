#!/usr/bin/env bash
# Task runner script

set -e

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <task_name>"
    echo ""
    echo "Use '$0 --list' to see available tasks"
    python -m azl_bot.tasks --list
    exit 1
fi

TASK_NAME="$1"

# Handle list command
if [ "$TASK_NAME" = "--list" ] || [ "$TASK_NAME" = "-l" ] || [ "$TASK_NAME" = "list" ]; then
    python -m azl_bot.tasks --list
    exit 0
fi

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

# Run the task using the new CLI interface
python -m azl_bot.tasks "$TASK_NAME"