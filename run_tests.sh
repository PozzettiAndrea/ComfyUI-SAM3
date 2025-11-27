#!/bin/bash

# Path to ComfyUI source code inside the App bundle
COMFY_PATH="/Applications/ComfyUI.app/Contents/Resources/ComfyUI"

# Path to your virtual environment python
PYTHON_EXEC="/Users/provos/Documents/ComfyUI/.venv/bin/python"

# Export PYTHONPATH to include ComfyUI source and current directory
export PYTHONPATH="$COMFY_PATH:$(pwd)"

# Check if HF_TOKEN is set (required for integration tests that download models)
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN is not set. Integration tests requiring model downloads will be skipped."
    echo "To run them, use: HF_TOKEN=your_token ./run_tests.sh"
fi

# Run pytest using the venv python
echo "Running tests with ComfyUI path: $COMFY_PATH"
"$PYTHON_EXEC" -m pytest "$@"
