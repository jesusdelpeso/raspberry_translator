#!/usr/bin/env python3
"""
Test script to verify installation and check audio devices
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    required_packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "sounddevice": "SoundDevice",
        "numpy": "NumPy",
        "scipy": "SciPy",
    }

    failed = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            failed.append(package)

    return len(failed) == 0


def test_audio_devices():
    """Test audio input/output devices"""
    print("\nTesting audio devices...")
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        print(f"\nFound {len(devices)} audio devices:")
        print("-" * 80)
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']}")
            print(f"   Max input channels: {device['max_input_channels']}")
            print(f"   Max output channels: {device['max_output_channels']}")
            print(f"   Default sample rate: {device['default_samplerate']}")
            print()

        default_input = sd.query_devices(kind="input")
        default_output = sd.query_devices(kind="output")

        print(f"Default input device: {default_input['name']}")
        print(f"Default output device: {default_output['name']}")

        return True
    except Exception as e:
        print(f"✗ Error testing audio: {e}")
        return False


def test_microphone():
    """Test microphone recording"""
    print("\nTesting microphone recording...")
    try:
        import sounddevice as sd
        import numpy as np

        print("Recording 2 seconds of audio...")
        duration = 2  # seconds
        sample_rate = 16000

        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
        )
        sd.wait()

        # Check if we got valid audio
        max_amplitude = np.max(np.abs(recording))
        print(f"✓ Recording successful!")
        print(f"  Max amplitude: {max_amplitude:.4f}")

        if max_amplitude < 0.001:
            print("  ⚠ Warning: Very quiet audio. Check microphone levels.")

        return True
    except Exception as e:
        print(f"✗ Error recording audio: {e}")
        return False


def test_pytorch():
    """Test PyTorch installation and device"""
    print("\nTesting PyTorch...")
    try:
        import torch

        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
        else:
            print("  Using CPU")

        # Test tensor operation
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.mm(x, y)
        print("  ✓ Basic tensor operations working")

        return True
    except Exception as e:
        print(f"✗ Error testing PyTorch: {e}")
        return False


def test_transformers():
    """Test Transformers library"""
    print("\nTesting Transformers library...")
    try:
        from transformers import __version__ as transformers_version

        print(f"✓ Transformers version: {transformers_version}")
        return True
    except Exception as e:
        print(f"✗ Error testing Transformers: {e}")
        return False


def main():
    print("=" * 80)
    print("Real-time Translator - System Test")
    print("=" * 80)
    print()

    results = []

    # Run all tests
    results.append(("Imports", test_imports()))
    results.append(("PyTorch", test_pytorch()))
    results.append(("Transformers", test_transformers()))
    results.append(("Audio Devices", test_audio_devices()))

    # Ask before microphone test
    print("\n" + "=" * 80)
    response = input("Test microphone recording? (y/n): ")
    if response.lower() == "y":
        results.append(("Microphone", test_microphone()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n✓ All tests passed! System is ready.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
