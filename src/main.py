#!/usr/bin/env python3
"""
Real-time Audio Translator for Raspberry Pi 5
Main entry point for the application
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import Config
from .lang_select import is_interactive, prompt_language_pair
from .logging_setup import setup_logging
from .recovery import RetryError
from .translator import RealTimeTranslator
from .validator import ConfigError, print_validation_report, validate_config

logger = logging.getLogger(__name__)


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
    parser.add_argument(
        "--bidirectional",
        "-b",
        action="store_true",
        default=None,
        help="Enable bidirectional / conversation mode (A\u2194B translation)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        default=False,
        help="Skip interactive language selection even when no languages are configured",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Validate configuration and exit (non-zero exit code on errors)",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        default=False,
        help="Launch the web UI instead of the terminal translator",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address for the web UI server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the web UI server (default: 7860)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Set logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Append log output to this file in addition to stderr",
    )
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()

    # Bootstrap logging early with CLI overrides (config may refine later)
    _early_level = args.log_level or "INFO"
    _early_file = args.log_file
    setup_logging(level=_early_level, log_file=_early_file)

    # Determine config file path
    config_path = args.config
    if not config_path:
        # Look for config.yaml in config folder
        default_config = Path(__file__).parent.parent / "config" / "config.yaml"
        if default_config.exists():
            config_path = str(default_config)
            logger.info("Using configuration file: %s", config_path)

    # Track whether a real config file was loaded (affects selector trigger)
    config_from_file = bool(config_path)

    # Load configuration
    if config_path:
        try:
            config = Config.from_yaml(config_path)
        except Exception as e:
            logger.error("Error loading config file: %s", e)
            logger.warning("Using default configuration.")
            config = Config()
            config_from_file = False
    else:
        config = Config()

    # Override with command line arguments if provided
    langs_from_cli = bool(args.source_lang or args.target_lang)
    if args.source_lang:
        config.source_lang = args.source_lang
    if args.target_lang:
        config.target_lang = args.target_lang
    if args.duration:
        config.recording_duration = args.duration
    if args.bidirectional:
        config.bidirectional_mode = True
    if args.no_interactive:
        config.interactive_lang_select = False

    # Re-apply logging with final config values (CLI flags take precedence)
    if args.log_level is None and args.log_file is None:
        # Reconfigure using values from config file
        setup_logging(level=config.log_level, log_file=config.log_file)
    elif args.log_level is None:
        setup_logging(level=config.log_level, log_file=args.log_file)
    elif args.log_file is None:
        setup_logging(level=args.log_level, log_file=config.log_file)
    # else: already configured above with both CLI values

    # ── Interactive language selection ────────────────────────────────────────
    # Trigger when:  no config file loaded, no explicit CLI lang args, and
    # interactive_lang_select is True (also requires stdin to be a TTY —
    # prompt_language_pair handles that check internally).
    if (
        not config_from_file
        and not langs_from_cli
        and config.interactive_lang_select
        and is_interactive()
    ):
        config.source_lang, config.target_lang = prompt_language_pair(
            config.source_lang, config.target_lang
        )

    # ── Configuration validation ──────────────────────────────────────────────
    validation = validate_config(config)
    print_validation_report(validation)
    if not validation.ok:
        print("\nFix the errors above before starting the translator.")
        return 2
    if args.validate:
        return 0  # Validate-only mode: exit after printing the report

    # Save config if requested
    if args.save_config:
        config.save_yaml(args.save_config)
        logger.info("Configuration saved. Exiting.")
        return 0

    # ── Web UI mode ───────────────────────────────────────────────────────────
    if args.web:
        try:
            import uvicorn
            from .web import build_app
        except ImportError:
            logger.critical(
                "Web UI requires 'fastapi' and 'uvicorn'. "
                "Install with: pip install fastapi uvicorn[standard]"
            )
            return 1

        app = build_app(config)
        logger.info("Starting web UI at http://%s:%d", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    # Create and start translator
    try:
        translator = RealTimeTranslator(config)
    except RuntimeError as exc:
        logger.critical("Startup failed: %s", exc)
        logger.critical("Check your internet connection and available disk space, then retry.")
        return 1

    translator.start_listening()

    return 0


if __name__ == "__main__":
    sys.exit(main())
