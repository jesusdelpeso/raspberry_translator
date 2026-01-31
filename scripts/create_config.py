#!/usr/bin/env python3
"""
Utility to generate a custom configuration file
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config, LANGUAGE_CODES


def print_language_options():
    """Print available language codes"""
    print("\nAvailable Language Codes:")
    print("=" * 60)
    for name, code in sorted(LANGUAGE_CODES.items()):
        print(f"  {name:25s} : {code}")
    print("=" * 60)


def interactive_config():
    """Create configuration interactively"""
    print("=" * 60)
    print("Real-time Translator - Configuration Generator")
    print("=" * 60)
    
    print_language_options()
    
    print("\nCurrent defaults:")
    print("  STT Model: openai/whisper-small")
    print("  Translation Model: facebook/nllb-200-distilled-1.3B")
    print("  TTS Model: facebook/mms-tts-eng")
    print("  Source Language: eng_Latn (English)")
    print("  Target Language: spa_Latn (Spanish)")
    print("  Recording Duration: 5 seconds")
    
    print("\n" + "=" * 60)
    print("Press Enter to keep default values")
    print("=" * 60)
    
    # Get user input with defaults
    source_lang = input("\nSource language code [eng_Latn]: ").strip() or "eng_Latn"
    target_lang = input("Target language code [spa_Latn]: ").strip() or "spa_Latn"
    
    duration_str = input("Recording duration in seconds [5]: ").strip()
    duration = int(duration_str) if duration_str else 5
    
    stt_model = input("\nSTT Model [openai/whisper-small]: ").strip() or "openai/whisper-small"
    trans_model = input("Translation Model [facebook/nllb-200-distilled-1.3B]: ").strip() or "facebook/nllb-200-distilled-1.3B"
    tts_model = input("TTS Model [facebook/mms-tts-eng]: ").strip() or "facebook/mms-tts-eng"
    
    use_gpu_str = input("\nUse GPU if available? [n]: ").strip().lower()
    use_gpu = use_gpu_str in ['y', 'yes', 'true']
    
    # Create config
    config = Config(
        stt_model=stt_model,
        translation_model=trans_model,
        tts_model=tts_model,
        source_lang=source_lang,
        target_lang=target_lang,
        recording_duration=duration,
        use_gpu=use_gpu,
    )
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate a configuration file for the translator"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="config/config.yaml",
        help="Output configuration file path (default: config/config.yaml)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Create configuration interactively",
    )
    parser.add_argument(
        "--list-languages",
        "-l",
        action="store_true",
        help="List available language codes and exit",
    )
    
    args = parser.parse_args()
    
    if args.list_languages:
        print_language_options()
        return 0
    
    if args.interactive:
        config = interactive_config()
    else:
        # Use defaults
        config = Config()
    
    # Save configuration
    config.save_yaml(args.output)
    print(f"\n✓ Configuration file created: {args.output}")
    print(f"\nTo use this configuration:")
    print(f"  python scripts/run_translator.py --config {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
