#!/bin/bash
# Convenience script to run streaming transcription
# Configuration is read from config/streaming_config.yaml by default
# Command line arguments can override config file settings
#
# Usage:
#   ./run_streaming_transcribe.sh                          # Use default config
#   ./run_streaming_transcribe.sh -c myconfig.yaml        # Use custom config
#   ./run_streaming_transcribe.sh --config myconfig.yaml  # Use custom config
#   ./run_streaming_transcribe.sh --language es           # Override language

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if it exists
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# Parse arguments to find config file specification
CONFIG_FILE=""
REMAINING_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Streaming Transcription - Usage:"
            echo ""
            echo "  $0 [options]"
            echo ""
            echo "Options:"
            echo "  -c, --config FILE    Use specified configuration file"
            echo "  -h, --help           Show this help message"
            echo "  --language LANG      Override language setting (en, es, fr, etc.)"
            echo "  --model MODEL        Override model setting"
            echo "  --list-devices       List available audio devices"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use default config"
            echo "  $0 -c myconfig.yaml                   # Use custom config"
            echo "  $0 --config config/parakeet.yaml      # Use Parakeet config"
            echo "  $0 --language es                      # Override to Spanish"
            echo ""
            echo "Configuration files are located in: $PROJECT_DIR/config/"
            exit 0
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

# If config file specified, check if it needs full path
if [ -n "$CONFIG_FILE" ]; then
    # If it's not an absolute path and doesn't start with ./, check in config directory
    if [[ "$CONFIG_FILE" != /* ]] && [[ "$CONFIG_FILE" != ./* ]]; then
        # Check if file exists as-is
        if [ ! -f "$CONFIG_FILE" ]; then
            # Try in config directory
            if [ -f "$PROJECT_DIR/config/$CONFIG_FILE" ]; then
                CONFIG_FILE="$PROJECT_DIR/config/$CONFIG_FILE"
            fi
        fi
    fi
    
    # Verify the config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Configuration file not found: $CONFIG_FILE"
        echo ""
        echo "Available configs in $PROJECT_DIR/config/:"
        ls -1 "$PROJECT_DIR/config/"*.yaml 2>/dev/null | xargs -n 1 basename
        exit 1
    fi
    
    echo "Using configuration from: $CONFIG_FILE"
    echo "(Command line arguments will override config file settings)"
    echo ""
    
    # Add --config to the arguments for Python script
    REMAINING_ARGS=("--config" "$CONFIG_FILE" "${REMAINING_ARGS[@]}")
else
    # Check if default config file exists and inform user
    DEFAULT_CONFIG="$PROJECT_DIR/config/streaming_config.yaml"
    if [ -f "$DEFAULT_CONFIG" ]; then
        echo "Using configuration from: $DEFAULT_CONFIG"
        echo "(Command line arguments will override config file settings)"
        echo ""
    fi
fi

# Run the streaming transcription script with remaining arguments
python3 "$SCRIPT_DIR/streaming_transcribe.py" "${REMAINING_ARGS[@]}"
