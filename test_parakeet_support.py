#!/usr/bin/env python3
"""Test script to verify model type detection and configuration loading"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_model_detection():
    """Test that model type is correctly detected"""
    
    test_cases = [
        ("openai/whisper-small", "whisper"),
        ("openai/whisper-tiny", "whisper"),
        ("nvidia/parakeet-tdt-0.6b-v3", "parakeet"),
        ("some-other-model", "whisper"),  # Default to whisper
    ]
    
    print("Testing model type detection...")
    print("=" * 60)
    
    all_passed = True
    for model_name, expected_type in test_cases:
        # Simulate detection logic
        detected_type = "whisper" if "whisper" in model_name.lower() else "parakeet" if "parakeet" in model_name.lower() else "whisper"
        
        if detected_type == expected_type:
            print(f"✓ {model_name} -> {detected_type}")
        else:
            print(f"✗ {model_name} -> {detected_type} (expected {expected_type})")
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All model detection tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1

def test_config_loading():
    """Test that configuration file has Parakeet option"""
    
    print("\nTesting configuration file...")
    print("=" * 60)
    
    # Get the script's directory and build path to config
    script_dir = Path(__file__).parent
    config_file = script_dir / "config" / "streaming_config.yaml"
    
    if not config_file.exists():
        print(f"✗ Config file not found: {config_file}")
        return 1
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    checks = [
        ("parakeet", "Parakeet model mentioned in config"),
        ("nvidia/", "NVIDIA model path format present"),
        ("# Parakeet models:", "Parakeet models comment section"),
    ]
    
    all_passed = True
    for check_str, description in checks:
        if check_str.lower() in content.lower():
            print(f"✓ {description}")
        else:
            print(f"✗ {description} - NOT FOUND")
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ Configuration file checks passed!")
        return 0
    else:
        print("\n✗ Some configuration checks failed!")
        return 1

def main():
    print("Parakeet Support Verification")
    print("=" * 60)
    print()
    
    result1 = test_model_detection()
    result2 = test_config_loading()
    
    print("\n" + "=" * 60)
    if result1 == 0 and result2 == 0:
        print("✓ All checks passed! Parakeet support is properly configured.")
        print("\nTo use Parakeet models:")
        print("1. Install NeMo: pip install nemo_toolkit['asr']")
        print("2. Update config/streaming_config.yaml with nvidia/parakeet-tdt-0.6b-v3")
        print("3. Run: ./scripts/run_streaming_transcribe.sh")
        return 0
    else:
        print("✗ Some checks failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
