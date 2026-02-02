# Real-time Audio Translator for Raspberry Pi 5

<div align="center">

🎙️ **Speak in one language, hear it in another** 🔊

A Python application that provides real-time speech translation using state-of-the-art AI models from Hugging Face. Designed to run efficiently on Raspberry Pi 5 hardware.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-5-C51A4A.svg)](https://www.raspberrypi.com/)

</div>

---


## 📖 Overview

This project implements a complete real-time translation pipeline that:
1. **Listens** to your voice through a microphone
2. **Transcribes** speech to text using AI speech recognition
3. **Translates** the text to your target language
4. **Speaks** the translation through speakers

Perfect for language learning, international communication, or accessibility applications.

### ✨ Key Features

- 🎯 **Real-time Translation**: Processes audio in configurable chunks for near real-time results
- �️ **Streaming Transcription**: Sentence-by-sentence real-time audio transcription (NEW!)
- �🌍 **200+ Languages**: Supports translation between over 200 languages
- 🤖 **State-of-the-art AI**: Uses OpenAI Whisper, Meta NLLB, and Hugging Face TTS models
- 🍓 **Raspberry Pi Optimized**: Configured to run efficiently on Raspberry Pi 5 hardware
- ⚙️ **Highly Configurable**: YAML-based configuration for easy customization
- 🔧 **Modular Architecture**: Clean, maintainable code structure
- 📦 **Easy Setup**: Automated installation and setup scripts

### 🔄 Translation Pipeline

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐      ┌──────────────┐
│ Microphone  │ ──▶  │ Speech-to-   │ ──▶  │ Translation │ ──▶  │ Text-to-     │
│   Input     │      │    Text      │      │   Engine    │      │   Speech     │
└─────────────┘      └──────────────┘      └─────────────┘      └──────────────┘
                            ▲                                            │
                            │                                            ▼
                     OpenAI Whisper                              ┌──────────────┐
                                                                  │   Speaker    │
                            ▲                                    │   Output     │
                            │                                    └──────────────┘
                     Meta NLLB-200
```

### 🤖 Default AI Models

| Component | Model | Description |
|-----------|-------|-------------|
| **Speech-to-Text** | `openai/whisper-small` | Balanced accuracy and performance for Raspberry Pi |
| **Speech-to-Text (Alt)** | `nvidia/parakeet-tdt-0.6b-v3` | High accuracy multilingual (25 languages, optional) |
| **Translation** | `facebook/nllb-200-distilled-1.3B` | Supports 200+ languages with good quality |
| **Text-to-Speech** | `facebook/mms-tts-eng` | Lightweight, suitable for edge devices |

*All models are customizable via configuration file. See [PARAKEET_SUPPORT.md](PARAKEET_SUPPORT.md) for Parakeet setup.*

---

## 🆕 Recent Updates

### February 2, 2026
- 🔧 **FIXED: Audio Streaming** - Resolved critical PortAudio threading timeout errors on Linux
  - Changed from callback to blocking read mode for reliable audio capture
  - Added retry logic and improved error handling
  - See [AUDIO_STREAMING_FIXES.md](AUDIO_STREAMING_FIXES.md) for technical details
- ✨ **NEW: Parakeet Model Support** - Added support for NVIDIA Parakeet TDT models with superior accuracy
- ✨ **NEW: Configuration File** - Streaming transcription now uses YAML config for easier setup
- 🐛 **Fixed**: Crash on interrupt (Ctrl+C) in streaming transcription
- 📝 **Enhanced**: Configuration-based workflow with command line overrides
- 🎯 **Improved**: Support for both Whisper and Parakeet ASR models

### February 1, 2026
- ✨ **NEW: Language Parameter** - Added `--language` parameter to streaming transcription for better accuracy with specific languages
- 🐛 **Fixed**: Python command compatibility issue in `run_streaming_transcribe.sh`
- 📝 **Enhanced**: Comprehensive documentation for streaming transcription with 14+ language codes
- ✅ **Tested**: Spanish language transcription working successfully
- 🎯 **Improved**: Warning suppression for cleaner output

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Streaming Transcription](#-streaming-transcription-new)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Language Codes](#-language-codes)
- [Performance Tips](#-performance-tips)
- [Troubleshooting](#-troubleshooting)
- [Advanced Usage](#-advanced-usage)
- [Contributing](#-contributing)
- [License](#-license

---

## 💻 Requirements

### Hardware
- **Raspberry Pi 5** (4GB+ RAM recommended, 8GB ideal)
- **Microphone** (USB or 3.5mm)
- **Speaker** or headphones
- **microSD card** (32GB+ recommended)
- **Stable internet connection** (for first-time model downloads)

### Software
- **Raspberry Pi OS** (64-bit recommended)
- **Python 3.9+**
- ~5GB free space for AI models

---

## 🚀 Installation

### 1. System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies (including PortAudio for audio handling)
sudo apt install -y python3-pip python3-venv portaudio19-dev python3-pyaudio

# Install audio libraries
sudo apt install -y libsndfile1 ffmpeg
```

> **⚠️ Important**: The `portaudio19-dev` and `python3-pyaudio` packages are **required** for the `sounddevice` Python library to work. Without these, you'll get an `OSError: PortAudio library not found` error.

### 2. Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure the Application

Copy the example configuration and customize it:

```bash
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml with your preferred settings
```

### 4. Download Models (Optional)

Models will be automatically downloaded on first run, but you can pre-download them:

**For Whisper models (default):**
- Models download automatically on first use
- No additional setup required

**For Parakeet models (optional, higher accuracy):**
```bash
# Install NeMo toolkit
pip install -r requirements_parakeet.txt

# Download Parakeet model (~600MB)
python scripts/download_parakeet_model.py
```

See [PARAKEET_SUPPORT.md](PARAKEET_SUPPORT.md) for details.

---

## ⚡ Quick Start

1. **Test your setup**:
   ```bash
   source .venv/bin/activate
   python test_setup.py
   ```

2. **Run the translator**:
   ```bash
   ./scripts/run.sh
   ```

### Common Commands

| Command | Description |
|---------|-------------|
| `./scripts/run.sh` | Run with default config |
| `./scripts/run.sh --config config/my_config.yaml` | Use custom config |
| `./scripts/run.sh --source-lang eng_Latn --target-lang fra_Latn` | Override languages |
| `./scripts/run.sh --duration 3` | Set 3-second recording chunks |
| `./scripts/run.sh --save-config config/saved.yaml` | Save current settings |
| `./scripts/run.sh --help` | Show all options |

### Common Use Cases

**English to Spanish translation:**
```bash
# Edit config/config.yaml to set:
# source: eng_Latn, target: spa_Latn
./scripts/run.sh
```

**French to English with custom duration:**
```bash
./scripts/run.sh --source-lang fra_Latn --target-lang eng_Latn --duration 8
```

**Quick test with different models:**
```bash
# Edit config/config.yaml to change models, then:
./scripts/run.sh --duration 3
```

---

## 🎙️ Streaming Transcription (NEW!)

In addition to the translation pipeline, this project now includes a **real-time streaming transcription** tool that continuously captures audio and transcribes it sentence by sentence.

### Quick Start

```bash
# Basic usage (uses default config)
./scripts/run_streaming_transcribe.sh

# Use specific config file
./scripts/run_streaming_transcribe.sh -c streaming_config_parakeet.yaml

# With GPU acceleration
./scripts/run_streaming_transcribe.sh --gpu

# Specify language for better accuracy
./scripts/run_streaming_transcribe.sh --language en

# Adjust sensitivity
./scripts/run_streaming_transcribe.sh --vad-threshold 0.015
```

### Key Features

- ✅ **Continuous transcription**: Runs indefinitely until stopped
- ✅ **Sentence detection**: Automatically detects sentence boundaries using punctuation and silence
- ✅ **Voice Activity Detection**: Only transcribes when speech is detected
- ✅ **Configurable parameters**: Adjust VAD threshold, silence duration, chunk size, etc.
- ✅ **Multiple Whisper models**: Choose from tiny/base/small/medium/large

### Example Session

```bash
$ ./scripts/run_streaming_transcribe.sh

================================================================================
STREAMING AUDIO TRANSCRIPTION
================================================================================
Sample rate: 16000 Hz
Chunk duration: 3.0s
VAD threshold: 0.02
Silence duration for sentence end: 1.5s
================================================================================

Processing audio... Speak into the microphone.
Press Ctrl+C to stop.

--------------------------------------------------------------------------------

[Sentence 1]: Hello, this is a test of the streaming transcription system.
--------------------------------------------------------------------------------

[Sentence 2]: It works really well for real-time transcription.
--------------------------------------------------------------------------------
```

### Available Options

| Option | Default | Description |
|--------|---------|-------------|
| `-c`, `--config` | `config/streaming_config.yaml` | Configuration file to use |
| `--model` | `openai/whisper-small` | Whisper/Parakeet model to use |
| `--language` | `None` | Language code (e.g., 'en', 'es', 'fr'). Auto-detect if not specified |
| `--vad-threshold` | `0.02` | Voice activity detection threshold |
| `--silence-duration` | `1.5` | Silence duration to mark sentence end (seconds) |
| `--chunk-duration` | `3.0` | Audio chunk duration (seconds) |
| `--gpu` | `False` | Use GPU if available |
| `--list-devices` | - | List audio devices and exit |
| `-h`, `--help` | - | Show help message with usage examples |

### Documentation

For comprehensive documentation, see [STREAMING_TRANSCRIPTION.md](STREAMING_TRANSCRIPTION.md), which includes:
- Detailed usage examples
- Parameter tuning guide
- Troubleshooting tips
- Technical implementation details
- Performance optimization

---

## 📁 Project Structure

```
raspberry_translator/
├── config/                          # Configuration files
│   ├── config.example.yaml         # Example configuration (committed)
│   └── config.yaml                 # User configuration
│
├── src/                            # Source code
│   ├── __init__.py                # Package initialization
│   ├── main.py                    # Application entry point
│   ├── config.py                  # Configuration management
│   ├── models.py                  # AI model loading
│   ├── audio_handler.py           # Audio I/O operations
│   └── translator.py              # Translation pipeline
│
├── scripts/                        # Executable scripts
│   ├── run_translator.py          # Main executable script
│   ├── run.sh                     # Convenience shell runner
│   ├── streaming_transcribe.py    # Streaming transcription app (NEW)
│   ├── run_streaming_transcribe.sh # Streaming transcription runner (NEW)
│   ├── setup.sh                   # Automated setup script
│   ├── test_setup.py              # System verification tests
│   ├── download_models.py         # Pre-download AI models
│   ├── create_config.py           # Configuration generator
│   └── language_selector.py       # Interactive language picker
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── CHANGELOG.md                   # Version history and changes (NEW)
├── STREAMING_TRANSCRIPTION.md     # Streaming transcription docs (NEW)
├── STREAMING_TRANSCRIPTION_FIXES.md # Testing & troubleshooting (NEW)
└── .gitignore                     # Git ignore rules
```

### Module Descriptions

| Module | Purpose |
|--------|---------|
| **src/main.py** | CLI argument parsing and application bootstrapping |
| **src/config.py** | Configuration loading, validation, and persistence |
| **src/models.py** | Hugging Face model initialization and management |
| **src/audio_handler.py** | Microphone input and speaker output handling |
| **src/translator.py** | Orchestrates the full translation pipeline |
| **scripts/streaming_transcribe.py** | Real-time streaming transcription with sentence detection |

---

## ⬇️ Download Models (Optional)

```bash
python scripts/download_models.py --all
```

This downloads all required models (~5GB) ahead of time to avoid delays during first use.

---

## Usage

### Basic Usage

The application uses a configuration file (`config/config.yaml`) to customize all settings:

```bash
./scripts/run.sh
```

### Using Configuration File

1. **Edit the config file** to customize settings:
   - Models to use (STT, Translation, TTS)
   - Source and target languages
   - Audio settings
   - Performance options

2. **Or create a custom config interactively**:
   ```bash
   python create_config.py --interactive --output config/my_config.yaml
   ```

3. **Use your custom configuration**:
   ```bash
   ./scripts/run.sh --config config/my_config.yaml
   ```

### Command Line Overrides

```bash
# Override languages
./scripts/run.sh --source-lang eng_Latn --target-lang fra_Latn

# Override recording duration
./scripts/run.sh --duration 3

# Use custom config with overrides
./scripts/run.sh --config config/my_config.yaml --duration 10
```

### List Available Languages

```bash
python create_config.py --list-languages
```

---

## 🌍 Language Codes

The translator uses NLLB language codes. Common ones:

| Language | Code |
|----------|------|
| English | `eng_Latn` |
| Spanish | `spa_Latn` |
| French | `fra_Latn` |
| German | `deu_Latn` |

**200+ languages supported.** Full list: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

---

## 🚀 Performance Tips

### For Raspberry Pi 5

1. **Model Selection** (edit `config/config.yaml`):
   - **Fastest**: Use `openai/whisper-tiny` for STT
   - **Balanced**: Use `openai/whisper-small` (default)
   - **Best Quality**: Use `openai/whisper-base` (slower)

2. **Memory Optimization**:
   - Set `low_memory: true` in config
   - Use shorter `recording_duration` (3-5 seconds)
   - Close other applications
   - Enable swap space if needed:
     ```bash
     sudo dphys-swapfile swapoff
     sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=2048
     sudo dphys-swapfile setup
     sudo dphys-swapfile swapon
     ```

---

## 🛠 Troubleshooting

### ⚠️ PortAudio Library Not Found

**Problem**: Application crashes with `OSError: PortAudio library not found`

**Solution**: Install the required system libraries:
```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio
```

This error occurs when the `sounddevice` Python package cannot find the PortAudio library. These system packages must be installed **before** running the application.

### 🔧 Translation Pipeline Error

**Problem**: Application crashes with `KeyError: "Invalid translation task translation, use 'translation_XX_to_YY' format"`

**Solution**: This has been fixed in the latest version. If you encounter this error:
1. Make sure you have the latest code: `git pull`
2. The fix removes the explicit task parameter from the translation pipeline, allowing it to infer from the model
3. The corrected code is in `src/models.py` line 73-79

### 🎤 Audio Not Working

**Problem**: No audio detected or microphone not found

**Solutions**:
```bash
# List available audio devices
python scripts/test_setup.py

# Test microphone manually
python -c "import sounddevice as sd; print(sd.query_devices())"

# Check ALSA configuration
arecord -l
alsamixer  # Adjust mic levels
```

### 💾 Out of Memory Errors

**Problem**: Application crashes with memory errors

**Solutions**:
1. Use smaller model: Edit `config/config.yaml`, set `stt_model: "openai/whisper-tiny"`
2. Reduce recording duration: `recording_duration: 3`
3. Lower batch size: `batch_size: 4`
4. Enable swap space (see Performance Tips above)
5. Ensure 8GB RAM Raspberry Pi for best experience

---

## 🎓 Advanced Usage

### Speech-to-Text Options
| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| `openai/whisper-tiny` | 39M | ⚡⚡⚡ | ⭐⭐ | Testing, fast response |
| `openai/whisper-base` | 74M | ⚡⚡ | ⭐⭐⭐ | Balanced |
| `openai/whisper-small` | 244M | ⚡ | ⭐⭐⭐⭐ | Production (default) |

### Text-to-Speech Options
| Model | Quality | Languages | Notes |
|-------|---------|-----------|-------|
| `facebook/mms-tts-eng` | Good | English | Lightweight (default) |

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report Bugs**: Open an issue describing the problem
2. **Suggest Features**: Share your ideas in the issues
3. **Submit PRs**:
   ```bash
   # Fork the repo, then:
   git clone https://github.com/yourusername/raspberry_translator.git
   cd raspberry_translator
   git checkout -b feature/your-feature
   # Make changes, test, commit
   git push origin feature/your-feature
   # Open a Pull Request
   ```

4. **Improve Documentation**: Help make this README even better
5. **Share Configurations**: Share your optimized configs for different use cases

## � Fixes and Updates

### Version History

#### February 1, 2026
**Fixed Issues:**

1. **Translation Pipeline Initialization Error**
   - **Problem**: Pipeline creation failed with error "Invalid translation task translation, use 'translation_XX_to_YY' format"
   - **Root Cause**: When passing a model object to the `pipeline()` function, transformers library couldn't automatically infer the task type and required explicit task specification. However, NLLB models don't support the standard task format.
   - **Solution**: Implemented a custom `TranslationWrapper` class that directly uses the model's `generate()` method instead of relying on the transformers pipeline. This provides better control over the translation process and properly handles NLLB's language token requirements.
   - **File Modified**: [src/models.py](src/models.py)
   - **Code Changes**:
     - Replaced pipeline-based translation with custom wrapper class
     - Properly sets `forced_bos_token_id` for NLLB language specification
     - Maintains pipeline-compatible return format for compatibility

2. **Audio Device Not Available**
   - **Problem**: `PortAudioError: Error starting stream: Wait timed out [PaErrorCode -9987]` when no microphone is connected
   - **Context**: This is expected behavior on systems without audio input devices
   - **Workaround**: Created [test_translation_simple.py](test_translation_simple.py) for testing translation pipeline without microphone requirement
   - **Future Enhancement**: Consider adding a mock audio input mode for development/testing

### Testing Without Microphone

To test the translation pipeline without a microphone:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the simple test
python test_translation_simple.py
```

**Test Output (Successful):**
```
============================================================
All models loaded successfully!
============================================================

Testing translation with sample text...
Input text: Hello, how are you today?

Translating...
Translated text: Hola, ¿cómo estás hoy?

✓ Translation successful!

Generating speech from translated text...
Generated audio: shape=(28416,), rate=16000Hz
✓ Speech generation successful!

============================================================
All tests passed! The translation pipeline is working.
============================================================
```

This test:
- ✓ Loads all three models (STT, Translation, TTS)
- ✓ Tests translation with sample text (English → Spanish)
- ✓ Generates speech output from translated text
- ✓ Verifies the complete pipeline works end-to-end

---

## 📊 Quick Reference

### Application Comparison

| Feature | Translation App | Streaming Transcription |
|---------|----------------|------------------------|
| **Purpose** | Translate speech between languages | Transcribe speech to text |
| **Output** | Spoken translation | Text transcription |
| **Mode** | Fixed-duration chunks | Continuous streaming |
| **Languages** | 200+ translation pairs | Auto-detect or specify |
| **Sentence Detection** | No | Yes (punctuation + silence) |
| **Main Script** | `./scripts/run.sh` | `./scripts/run_streaming_transcribe.sh` |
| **Use Cases** | International communication, learning | Transcription, captions, note-taking |

### Command Cheat Sheet

```bash
# Translation Application
./scripts/run.sh                           # Default translation
./scripts/run.sh --duration 8              # 8-second recording chunks
./scripts/run.sh --source-lang fra_Latn \
  --target-lang eng_Latn                   # French to English

# Streaming Transcription
./scripts/run_streaming_transcribe.sh                     # Auto-detect language
./scripts/run_streaming_transcribe.sh --language en      # English transcription
./scripts/run_streaming_transcribe.sh --language es      # Spanish transcription
./scripts/run_streaming_transcribe.sh --gpu              # Use GPU acceleration
./scripts/run_streaming_transcribe.sh --vad-threshold 0.015  # Adjust sensitivity
```

---

## 📝 License

This project is open source and available under the MIT License.

**Model Licenses:**
- OpenAI Whisper: MIT License
- Meta NLLB: CC-BY-NC 4.0
- Hugging Face Models: Check individual model cards for specific terms

## 🙏 Acknowledgments

This project is made possible by:

- **[OpenAI](https://openai.com/)** for the Whisper speech recognition model
- **[Meta AI](https://ai.meta.com/)** for the NLLB translation model
- **[Hugging Face](https://huggingface.co/)** for the Transformers library and model hosting
- **[Raspberry Pi Foundation](https://www.raspberrypi.org/)** for accessible computing hardware
- The open-source community for continuous improvements

---

## 📚 Additional Documentation

- [STREAMING_TRANSCRIPTION.md](STREAMING_TRANSCRIPTION.md) - Complete streaming transcription guide
- [STREAMING_TRANSCRIPTION_FIXES.md](STREAMING_TRANSCRIPTION_FIXES.md) - Testing results and troubleshooting
- [CHANGELOG.md](CHANGELOG.md) - Version history and recent changes
- [FIXES_SUMMARY.md](FIXES_SUMMARY.md) - Historical bug fixes and solutions

---

**Version**: 1.1.0 (February 2026)  
**Status**: Active Development  
**Python**: 3.9+  
**Platform**: Raspberry Pi 5 (optimized), Linux (tested)
