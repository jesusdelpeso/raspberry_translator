#!/usr/bin/env python3
"""
Utility script to download and cache models before first run
This is useful to avoid delays during the first translation
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)

from src.config import Config


def download_model(model_name: str, task: str):
    """Download and cache a model"""
    print(f"\nDownloading {task} model: {model_name}")
    print("-" * 80)

    try:
        if task == "stt":
            # Download STT model
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name, low_cpu_mem_usage=True, use_safetensors=True
            )
            processor = AutoProcessor.from_pretrained(model_name)
            print(f"✓ Successfully downloaded {model_name}")

        elif task == "translation":
            # Download translation model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            pipe = pipeline("translation", model=model_name, tokenizer=tokenizer)
            print(f"✓ Successfully downloaded {model_name}")

        elif task == "tts":
            # Download TTS model
            pipe = pipeline("text-to-speech", model=model_name)
            print(f"✓ Successfully downloaded {model_name}")

        else:
            print(f"✗ Unknown task: {task}")
            return False

        return True

    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and cache models for the translator"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models (STT, Translation, TTS)",
    )
    parser.add_argument("--stt", action="store_true", help="Download STT model only")
    parser.add_argument(
        "--translation", action="store_true", help="Download translation model only"
    )
    parser.add_argument("--tts", action="store_true", help="Download TTS model only")

    args = parser.parse_args()

    # Load config
    config = Config()

    print("=" * 80)
    print("Model Download Utility")
    print("=" * 80)
    print("\nThis will download models to ~/.cache/huggingface/")
    print("Models may be several GB in size. Ensure you have enough space.")
    print()

    # Determine which models to download
    download_stt = args.all or args.stt
    download_translation = args.all or args.translation
    download_tts = args.all or args.tts

    # If no specific args, download all
    if not (download_stt or download_translation or download_tts):
        download_stt = download_translation = download_tts = True

    results = []

    if download_stt:
        results.append(("STT", download_model(config.stt_model, "stt")))

    if download_translation:
        results.append(
            ("Translation", download_model(config.translation_model, "translation"))
        )

    if download_tts:
        results.append(("TTS", download_model(config.tts_model, "tts")))

    # Summary
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)

    for model_type, success in results:
        status = "✓ Success" if success else "✗ Failed"
        print(f"{model_type:15s}: {status}")

    all_success = all(result[1] for result in results)

    if all_success:
        print("\n✓ All models downloaded successfully!")
        print("You can now run the translator without delays.")
        return 0
    else:
        print("\n✗ Some downloads failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
