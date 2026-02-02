#!/usr/bin/env python3
"""Test script to verify Parakeet model loading and transcription."""

import os
import sys

# Set environment variables
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Testing Parakeet Model Integration")
print("=" * 80)

# Test 1: Import NeMo
print("\n1. Testing NeMo import...")
try:
    import nemo.collections.asr as nemo_asr
    print("   ✓ NeMo ASR imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import NeMo: {e}")
    sys.exit(1)

# Test 2: Load Parakeet model
print("\n2. Loading Parakeet model...")
try:
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v3"
    )
    print(f"   ✓ Model loaded successfully")
    print(f"   Model type: {type(model).__name__}")
except Exception as e:
    print(f"   ✗ Failed to load model: {e}")
    sys.exit(1)

# Test 3: Create a simple audio file for testing
print("\n3. Testing transcription with dummy audio...")
try:
    import numpy as np
    import soundfile as sf
    import tempfile
    
    # Create a short silent audio file (1 second of silence)
    sample_rate = 16000
    duration = 1.0
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        sf.write(tmp_path, audio, sample_rate)
    
    # Transcribe
    result = model.transcribe([tmp_path])
    
    # Clean up
    os.unlink(tmp_path)
    
    print(f"   ✓ Transcription completed")
    print(f"   Result: {result}")
    
except Exception as e:
    print(f"   ✗ Transcription failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("All tests passed! Parakeet model is working correctly.")
print("=" * 80)
