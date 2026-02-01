#!/usr/bin/env python3
"""
Simple translation test without microphone requirement
Tests the translation pipeline with synthetic data
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.models import ModelLoader


def test_translation():
    """Test the translation pipeline"""
    print("=" * 60)
    print("Testing Translation Pipeline (No Microphone Required)")
    print("=" * 60)
    print()
    
    # Load default config
    config_path = Path(__file__).parent / "config" / "config.yaml"
    if config_path.exists():
        print(f"Loading configuration from: {config_path}")
        config = Config.from_yaml(str(config_path))
    else:
        print("Using default configuration")
        config = Config()
    
    print(f"Source language: {config.source_lang}")
    print(f"Target language: {config.target_lang}")
    print()
    
    # Initialize model loader
    print("Initializing models...")
    model_loader = ModelLoader(config)
    
    # Load STT model
    print("\n1. Loading Speech-to-Text model...")
    stt_pipe = model_loader.load_stt_model()
    print("   ✓ STT model loaded")
    
    # Load translation model  
    print("\n2. Loading Translation model...")
    translator = model_loader.load_translation_model()
    print("   ✓ Translation model loaded")
    
    # Load TTS model
    print("\n3. Loading Text-to-Speech model...")
    tts_pipe = model_loader.load_tts_model()
    print("   ✓ TTS model loaded")
    
    print("\n" + "=" * 60)
    print("All models loaded successfully!")
    print("=" * 60)
    print()
    
    # Test with sample text (skip audio recording)
    print("Testing translation with sample text...")
    test_text = "Hello, how are you today?"
    print(f"Input text: {test_text}")
    
    # Test translation
    print("\nTranslating...")
    try:
        translation_result = translator(test_text)
        translated_text = translation_result[0]["translation_text"]
        print(f"Translated text: {translated_text}")
        print("\n✓ Translation successful!")
    except Exception as e:
        print(f"\n✗ Translation error: {e}")
        return 1
    
    # Test TTS (generate audio but don't play)
    print("\nGenerating speech from translated text...")
    try:
        speech_output = tts_pipe(translated_text)
        audio_array = speech_output["audio"]
        sampling_rate = speech_output["sampling_rate"]
        print(f"Generated audio: shape={np.array(audio_array).shape}, rate={sampling_rate}Hz")
        print("✓ Speech generation successful!")
    except Exception as e:
        print(f"✗ Speech generation error: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("All tests passed! The translation pipeline is working.")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(test_translation())
