#!/usr/bin/env python3
"""
Download Parakeet model from Hugging Face.

This script downloads the NVIDIA Parakeet TDT model files needed for
high-accuracy multilingual transcription. The model is approximately 600MB.
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    from tqdm.auto import tqdm
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install requirements first: pip install -r requirements.txt")
    sys.exit(1)

def download_parakeet_model():
    """Download Parakeet TDT model from Hugging Face."""
    
    model_id = "nvidia/parakeet-tdt-0.6b-v3"
    filename = "parakeet-tdt-0.6b-v3.nemo"
    
    print("=" * 80)
    print("Parakeet Model Downloader")
    print("=" * 80)
    print(f"\nModel: {model_id}")
    print(f"File: {filename}")
    print(f"Size: ~600 MB")
    print()
    
    # Check if already downloaded
    try:
        cached_path = try_to_load_from_cache(model_id, filename)
        
        if cached_path and os.path.exists(cached_path):
            size_mb = os.path.getsize(cached_path) / (1024**2)
            print(f"✓ Model already downloaded!")
            print(f"  Location: {cached_path}")
            print(f"  Size: {size_mb:.2f} MB")
            print()
            return True
    except Exception as e:
        print(f"Could not check cache: {e}")
    
    # Download the model
    print("Downloading model from Hugging Face...")
    print("This will take several minutes depending on your connection.")
    print()
    
    try:
        # The download includes a progress bar from Hugging Face
        path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            force_download=False,
            resume_download=True
        )
        
        size_mb = os.path.getsize(path) / (1024**2)
        
        print()
        print("=" * 80)
        print("✓ Download complete!")
        print(f"  Location: {path}")
        print(f"  Size: {size_mb:.2f} MB")
        print("=" * 80)
        print()
        print("You can now use Parakeet with:")
        print("  ./scripts/run_streaming_transcribe.sh -c streaming_config_parakeet.yaml")
        print()
        
        return True
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        print("You can resume the download by running this script again.")
        return False
        
    except Exception as e:
        print(f"\n✗ Error during download: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Try again - downloads can be resumed")
        print("  3. Check Hugging Face status: https://status.huggingface.co/")
        return False

def main():
    """Main function."""
    
    # Check if NeMo is installed
    try:
        import nemo.collections.asr
        print("✓ NeMo toolkit is installed")
        print()
    except ImportError:
        print("✗ NeMo toolkit not found!")
        print()
        print("Please install NeMo first:")
        print("  pip install nemo_toolkit['asr']")
        print()
        print("Or use the requirements file:")
        print("  pip install -r requirements_parakeet.txt")
        print()
        sys.exit(1)
    
    success = download_parakeet_model()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
