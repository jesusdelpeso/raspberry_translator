#!/bin/bash
# Convenience shell script to run the translator
# Activates the virtual environment, sets up the PYTHONPATH and runs the translator

# Get the project root directory (parent of the scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# If conda is active, restart this script in a clean environment
if [ -n "$CONDA_PREFIX" ] && [ -z "$CONDA_ISOLATED" ]; then
    echo "Restarting in clean environment to avoid conda library conflicts..."
    # Use env -i to start with a clean environment, but preserve essential variables
    exec env -i \
        HOME="$HOME" \
        USER="$USER" \
        PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        CONDA_ISOLATED="1" \
        bash "$0" "$@"
fi

# Activate the virtual environment if it exists
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
    echo "Activated virtual environment at ${PROJECT_ROOT}/.venv"
else
    echo "Warning: Virtual environment not found at ${PROJECT_ROOT}/.venv"
    echo "Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
fi

# Add the project root to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Preload system libstdc++ to prevent conda's outdated version from being used
# This fixes the GLIBCXX version conflict
export LD_PRELOAD="/lib/x86_64-linux-gnu/libstdc++.so.6:${LD_PRELOAD}"

# Run the translator with all passed arguments
python3 "${SCRIPT_DIR}/run_translator.py" "$@"
