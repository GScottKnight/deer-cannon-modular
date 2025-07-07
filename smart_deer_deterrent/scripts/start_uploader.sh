#!/bin/bash

# Start the Azure upload service
# This script runs the uploader in the background

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
source "$PROJECT_DIR/venv/bin/activate"

# Change to project directory
cd "$PROJECT_DIR"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found!"
    echo "Copy .env.template to .env and add your Azure credentials"
    exit 1
fi

# Start the uploader
echo "Starting Azure upload service..."
python azure_uploader.py