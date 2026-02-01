#!/bin/bash
# Setup script for Raspberry Pi 5 Real-time Translator

set -e  # Exit on error

# Get the project root directory (parent of the scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Change to project root
cd "${PROJECT_ROOT}"

echo "=================================="
echo "Raspberry Pi Real-time Translator"
echo "Setup Script"
echo "=================================="
echo ""

# Check if running on Raspberry Pi
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model)
    echo "Detected: $MODEL"
else
    echo "Warning: Not detected as Raspberry Pi"
fi
echo ""

# Update system
echo "Step 1: Updating system packages..."
sudo apt update
echo ""

# Install system dependencies
echo "Step 2: Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    alsa-utils
echo ""

# Create virtual environment
echo "Step 3: Creating Python virtual environment..."
python3 -m venv .venv
echo ""

# Activate virtual environment
echo "Step 4: Activating virtual environment..."
source .venv/bin/activate
echo ""

# Upgrade pip
echo "Step 5: Upgrading pip..."
pip install --upgrade pip
echo ""

# Install Python dependencies
echo "Step 6: Installing Python dependencies (this may take a while)..."
pip install -r requirements.txt
echo ""

# Test audio setup
echo "Step 7: Testing audio setup..."
echo "Available audio devices:"
python3 -c "import sounddevice as sd; print(sd.query_devices())"
echo ""

# Create models directory
echo "Step 8: Creating models cache directory..."
mkdir -p ~/.cache/huggingface
echo ""

echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To use the translator:"
echo "1. Use the provided run script:"
echo "   ./scripts/run.sh"
echo ""
echo "Or manually:"
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Run the translator:"
echo "   ./scripts/run_translator.py"
echo ""
echo "For more options:"
echo "   ./scripts/run.sh --help"
echo ""
echo "Note: First run will download models (may take time)"
echo "=================================="
