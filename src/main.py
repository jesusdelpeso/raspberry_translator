#!/usr/bin/env python3
"""
Real-time Audio Translator for Raspberry Pi 5
Main entry point for the application
"""

import argparse
import sys
from pathlib import Path

from .config import Config
from .translator import RealTimeTranslator


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Real-time Audio Translator for Raspberry Pi"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to configuration YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        help="Source language code (overrides config file)",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        help="Target language code (overrides config file)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Recording duration in seconds per chunk (overrides config file)",
    )
    parser.add_argument(
        "--save-config",
        type=str,
        help="Save current configuration to specified file and exit",
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()

    # Determine config file path
    config_path = args.config
    if not config_path:
        # Look for config.yaml in config folder
        default_config = Path(__file__).parent.parent / "config" / "config.yaml"
        if default_config.exists():
            config_path = str(default_config)
            print(f"Using configuration file: {config_path}")

    # Load configuration
    if config_path:
        try:
            config = Config.from_yaml(config_path)
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration.")
            config = Config()
    else:
        config = Config()

    # Override with command line arguments if provided
    if args.source_lang:
        config.source_lang = args.source_lang
    if args.target_lang:
        config.target_lang = args.target_lang
    if args.duration:
        config.recording_duration = args.duration

    # Save config if requested
    if args.save_config:
        config.save_yaml(args.save_config)
        print("Configuration saved. Exiting.")
        return 0

    # Create and start translator
    translator = RealTimeTranslator(config)
    translator.start_listening()

    return 0


if __name__ == "__main__":
    sys.exit(main())
