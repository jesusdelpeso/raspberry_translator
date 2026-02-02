# Parakeet Model Integration - Fix Documentation

**Date:** February 2, 2026

## Issue Found

When running the streaming transcription script with the Parakeet configuration:

```bash
./scripts/run_streaming_transcribe.sh -c streaming_config_parakeet.yaml
```

**Problems encountered:**

1. **Missing NeMo Toolkit** - The NVIDIA NeMo toolkit was not installed
2. **Model Not Downloaded** - The Parakeet model (~600MB) needs to be downloaded from Hugging Face before first use

## Root Cause

The Parakeet model requires additional setup compared to Whisper models:
- **NeMo ASR package** is needed to load and run Parakeet models
- **Large model file** (600MB) must be downloaded from Hugging Face Hub
- First-time model loading can take several minutes depending on internet connection

## Fixes Applied

### 1. Installed NeMo Toolkit

Added the NeMo ASR package to support Parakeet models:

```bash
pip install nemo_toolkit['asr']
```

The installation includes:
- `nemo_toolkit[asr]>=2.0.0` - Core NeMo ASR functionality
- Dependencies for audio processing and model loading

### 2. Created Model Download Helper Script

Created `/scripts/download_parakeet_model.py` to handle model downloads:

**Features:**
- Checks if model is already downloaded
- Shows download progress
- Supports resume if interrupted
- Provides clear error messages
- Verifies NeMo installation

**Usage:**
```bash
cd /home/wotan/explorations/raspberry_translator
source .venv/bin/activate
python scripts/download_parakeet_model.py
```

### 3. Updated Documentation

Enhanced documentation across multiple files to clarify:
- NeMo installation requirement
- Model download process
- Config file selection with `-c` parameter
- Troubleshooting steps

## How to Use Parakeet Now

### Step 1: Install NeMo (if not already installed)

```bash
cd /home/wotan/explorations/raspberry_translator
source .venv/bin/activate
pip install -r requirements_parakeet.txt
```

### Step 2: Download the Model

```bash
python scripts/download_parakeet_model.py
```

This will:
- Check if NeMo is installed
- Download the Parakeet model (~600MB)
- Show progress and confirm completion
- Support resume if interrupted

### Step 3: Run Streaming Transcription

```bash
# Using the dedicated Parakeet config
./scripts/run_streaming_transcribe.sh -c streaming_config_parakeet.yaml

# Or using command line override
./scripts/run_streaming_transcribe.sh --model nvidia/parakeet-tdt-0.6b-v3
```

## Verification

### Test NeMo Installation

```bash
python -c "import nemo.collections.asr as nemo_asr; print('NeMo ASR installed successfully')"
```

Expected output:
```
[NeMo W ...] Megatron num_microbatches_calculator not found, using Apex version.
NeMo ASR installed successfully
```

### Test Model Download

```bash
python scripts/download_parakeet_model.py
```

Expected output (if already downloaded):
```
✓ Model already downloaded!
  Location: /home/wotan/.cache/huggingface/hub/models--nvidia--parakeet-tdt-0.6b-v3/...
  Size: 600.XX MB
```

### Test Streaming Transcription

```bash
# List devices (quick test)
./scripts/run_streaming_transcribe.sh -c streaming_config_parakeet.yaml --list-devices
```

## Performance Notes

### Parakeet vs Whisper

| Aspect | Whisper Small | Parakeet TDT |
|--------|--------------|--------------|
| Model Size | ~244MB | ~600MB |
| Parameters | 244M | 600M |
| Languages | 100+ | 25 European |
| Download Time | ~30 seconds | ~2-5 minutes |
| First Load | ~10 seconds | ~20 seconds |
| Accuracy | Good | Superior (on supported languages) |
| Punctuation | Basic | Automatic with capitalization |
| Dependencies | transformers | nemo_toolkit |

### Supported Languages (Parakeet)

Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish, Russian, Ukrainian

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'nemo'`

**Solution:**
```bash
pip install nemo_toolkit['asr']
```

### Issue: Model download times out or fails

**Solutions:**
1. Run the download script again (it resumes automatically)
2. Check internet connection
3. Check Hugging Face status: https://status.huggingface.co/
4. Try downloading manually:
   ```bash
   python -c "from huggingface_hub import hf_hub_download; hf_hub_download('nvidia/parakeet-tdt-0.6b-v3', 'parakeet-tdt-0.6b-v3.nemo')"
   ```

### Issue: File lock error during download

**Solution:**
```bash
rm -rf ~/.cache/huggingface/hub/.locks/models--nvidia--parakeet-tdt-0.6b-v3/
python scripts/download_parakeet_model.py
```

### Issue: Slow transcription on first run

**Expected behavior:**
- First run downloads model config files
- Subsequent runs are faster
- Parakeet is slower than whisper-tiny but more accurate

## Files Modified/Created

1. **Created:** `/scripts/download_parakeet_model.py`
   - Helper script for model download
   - Includes progress tracking and error handling

2. **Updated:** `/CHANGELOG.md`
   - Added Parakeet model fix documentation
   - Added config file selection feature

3. **Updated:** `/README.md`
   - Added `-c`/`--config` parameter documentation
   - Enhanced options table

4. **Updated:** `/QUICKSTART_CONFIG.md`
   - Added config file selection examples
   - Added Parakeet config usage

5. **Updated:** `/config/README.md`
   - Enhanced config file path resolution docs

6. **Updated:** `/PARAKEET_SUPPORT.md`
   - Added pre-configured config file usage
   - Clarified installation steps

## Summary

**Before Fix:**
- Running Parakeet config failed with missing dependencies
- No clear process for model download
- Users had to figure out NeMo installation

**After Fix:**
- Clear installation process with helper script
- Automatic model download with progress tracking
- Resume support for interrupted downloads
- Comprehensive documentation
- Easy config file selection with `-c` parameter

**Status:** ✓ Resolved - Parakeet model support is now fully functional after completing the installation and download steps.
