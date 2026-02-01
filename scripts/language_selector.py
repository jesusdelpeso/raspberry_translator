#!/usr/bin/env python3
"""
Interactive language selector for easier language pair selection
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import LANGUAGE_CODES


def display_languages():
    """Display available languages"""
    print("\nAvailable Languages:")
    print("=" * 50)

    languages = list(LANGUAGE_CODES.items())
    for i, (name, code) in enumerate(languages, 1):
        print(f"{i:2d}. {name:25s} ({code})")

    print("=" * 50)
    return languages


def get_language_choice(prompt, languages):
    """Get user's language choice"""
    while True:
        try:
            print(f"\n{prompt}")
            choice = input("Enter number or language name: ").strip().lower()

            # Try to parse as number
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(languages):
                    return languages[idx][1]
            except ValueError:
                pass

            # Try to match language name
            for name, code in languages:
                if name.lower() == choice or choice in name.lower():
                    return code

            print("Invalid choice. Please try again.")

        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)


def main():
    print("=" * 50)
    print("Real-time Translator - Language Selector")
    print("=" * 50)

    languages = display_languages()

    source_lang = get_language_choice("Select source language:", languages)
    target_lang = get_language_choice("Select target language:", languages)

    # Find language names
    source_name = next(name for name, code in languages if code == source_lang)
    target_name = next(name for name, code in languages if code == target_lang)

    print("\n" + "=" * 50)
    print("Configuration:")
    print(f"  Source: {source_name} ({source_lang})")
    print(f"  Target: {target_name} ({target_lang})")
    print("=" * 50)

    print("\nTo run the translator with these settings:")
    print(
        f"python scripts/run_translator.py --source-lang {source_lang} --target-lang {target_lang}"
    )
    print("\nOr copy this command:")
    print("-" * 50)
    print(
        f"python scripts/run_translator.py --source-lang {source_lang} --target-lang {target_lang}"
    )
    print("-" * 50)


if __name__ == "__main__":
    main()
