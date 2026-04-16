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

- 🎯 **Real-time Translation**: VAD-driven streaming pipeline that segments audio at natural speech boundaries — no arbitrary time cuts
- 🌍 **200+ Languages**: Supports translation between over 200 languages via Meta NLLB-200
- 🤖 **State-of-the-art AI**: OpenAI Whisper (STT), Meta NLLB-200 (translation), Hugging Face MMS TTS (speech output)
- 🔊 **Auto Language Detection**: Whisper detects the spoken language each utterance — no need to configure `source_lang` manually
- 🎙 **Dynamic TTS**: TTS model is automatically chosen to match the target language (e.g. `spa_Latn` → `facebook/mms-tts-spa`)
- 🔁 **Bidirectional Mode**: Two speakers can hold a real-time conversation in different languages with automatic direction switching
- 📜 **Live Transcript**: Bordered conversation history printed to the terminal and optionally saved as JSONL
- 🌐 **Web UI**: Dark-mode browser dashboard with Start/Stop controls, live auto-scrolling transcript, and a REST + SSE API
- 🔁 **ONNX Runtime Acceleration**: Optional 2-4× faster CPU inference on Raspberry Pi via `optimum[onnxruntime]`
- ⚙️ **Highly Configurable**: YAML-based config with CLI overrides and interactive language picker at startup
- 🛡 **Graceful Recovery**: Auto-retry on model load failures and audio device reconnect with configurable backoff
- 📦 **Installable Package**: `pip install -e .` registers the `raspberry-translator` CLI command
- 🧪 **Full Test Suite**: 219 pytest tests across 7 test files with hardware mocks — runs offline

### 🔄 Translation Pipeline

```
┌──────────────┐    ┌─────────────────┐    ┌───────────────┐    ┌──────────────┐
│  Microphone  │─▶  │  VAD / Silero   │─▶  │ OpenAI Whisper│─▶  │  Meta NLLB   │
│    Input     │    │  (speech only)  │    │  STT + lang   │    │  Translation │
└──────────────┘    └─────────────────┘    │   detection   │    └──────┬───────┘
                                           └───────────────┘           │
                                                                        ▼
                    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
                    │   Speaker    │◀─  │  HF MMS TTS     │◀─  │  Translated  │
                    │   Output     │    │  (auto-selected) │    │    Text      │
                    └──────────────┘    └─────────────────┘    └──────────────┘
                                                  │
                             ┌────────────────────┴───────────────────┐
                             │         Web UI / Terminal transcript    │
                             └────────────────────────────────────────┘
```

### 🤖 Default AI Models

| Component | Model | Description |
|-----------|-------|-------------|
| **Speech-to-Text** | `openai/whisper-small` | Balanced accuracy and performance for Raspberry Pi |
| **Translation** | `facebook/nllb-200-distilled-1.3B` | Supports 200+ languages with good quality |
| **Text-to-Speech** | `facebook/mms-tts-{lang}` | Auto-selected from target language; lightweight, edge-device friendly |

*All models are customizable via the configuration file. Smaller Whisper variants (`whisper-tiny`, `whisper-base`) trade accuracy for speed.*

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Usage](#usage)
- [Project Structure](#-project-structure)
- [Language Codes](#-language-codes)
- [Performance Tips](#-performance-tips)
- [Troubleshooting](#-troubleshooting)
- [Advanced Usage](#-advanced-usage)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)

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

# Install system dependencies
sudo apt install -y python3-pip python3-venv portaudio19-dev python3-pyaudio

# Install audio libraries
sudo apt install -y libsndfile1 ffmpeg
```

### 2. Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Or install as an editable package (exposes the `raspberry-translator` CLI command)
pip install -e .
```

### 3. Configure the Application

Copy the example configuration and customize it:

```bash
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml with your preferred settings
```

### 4. Download Models (Optional)

Models will be automatically downloaded on first run, but you can pre-download them (~5GB):

```bash
source .venv/bin/activate
python scripts/download_models.py --all
```

---

## ⚡ Quick Start

1. **Test your setup**:
   ```bash
   source .venv/bin/activate
   python scripts/test_setup.py
   ```

2. **Run the translator**:
   ```bash
   # Via shell script
   ./scripts/run.sh

   # Or via the installed CLI command (after pip install -e .)
   raspberry-translator
   raspberry-translator --config config/my_config.yaml
   ```

### Common Commands

| Command | Description |
|---------|-------------|
| `./scripts/run.sh` | Run terminal translator with default config |
| `./scripts/run.sh -c config/my_config.yaml` | Use a custom config file |
| `./scripts/run.sh --source-lang eng_Latn --target-lang fra_Latn` | Override languages |
| `./scripts/run.sh --duration 3` | 3-second recording chunks (legacy fixed-duration mode) |
| `./scripts/run.sh -b` | Enable bidirectional / conversation mode |
| `./scripts/run.sh --validate` | Validate config and exit (non-zero on errors) |
| `./scripts/run.sh --save-config config/saved.yaml` | Save effective config to file |
| `./scripts/run.sh --no-interactive` | Skip startup language picker |
| `raspberry-translator --web` | Launch the web UI on port 7860 |
| `raspberry-translator --web --host 0.0.0.0 --port 8080` | Web UI on a custom port |
| `raspberry-translator --log-level DEBUG` | Verbose per-chunk diagnostics |
| `raspberry-translator --log-file logs/session.log` | Append logs to file |
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

**Web UI (live transcript in browser):**
```bash
# Install web dependencies (already in requirements.txt)
pip install fastapi uvicorn[standard]

# Launch (opens at http://localhost:7860)
raspberry-translator --web

# On a different port, accessible from the network
raspberry-translator --web --host 0.0.0.0 --port 8080
```
The web interface provides a dark-mode dashboard with:
- ▶ / ■ Start/Stop controls
- Live auto-scrolling transcript (via Server-Sent Events)
- Configuration overview panel
- REST API: `GET /api/status`, `GET /api/config`, `GET /api/history`, `GET /api/transcript` (SSE)

**Quick test with different models:**
```bash
# Edit config/config.yaml to change models, then:
./scripts/run.sh --duration 3
```

---

## 📁 Project Structure

```
raspberry_translator/
├── config/
│   ├── config.example.yaml         # Fully documented example (copy to config.yaml)
│   └── config.yaml                 # Your local config (gitignored)
│
├── src/                            # Application package
│   ├── __init__.py
│   ├── main.py                     # CLI entry point & argument parsing
│   ├── config.py                   # Config dataclass + YAML load/save
│   ├── models.py                   # Model loading (PyTorch & ONNX paths)
│   ├── audio_handler.py            # Microphone recording, speaker playback, VAD streaming
│   ├── translator.py               # Main pipeline orchestrator
│   ├── vad.py                      # Silero VAD wrapper (silence filtering)
│   ├── lang_detect.py              # Whisper language detection + ISO→NLLB mapping
│   ├── lang_select.py              # Interactive language picker (TTY only)
│   ├── tts_lang.py                 # NLLB code → MMS TTS model name mapping
│   ├── history.py                  # Conversation transcript + terminal display
│   ├── validator.py                # Config validation (errors + warnings)
│   ├── logging_setup.py            # Logging configuration (level, optional file)
│   ├── recovery.py                 # retry() helper + RetryError
│   └── web.py                      # FastAPI web UI (SSE transcript, REST API)
│
├── tests/
│   ├── conftest.py                 # Shared stubs + fixtures (runs offline)
│   ├── test_config.py              # Config dataclass, YAML round-trips (50 tests)
│   ├── test_audio_handler.py       # AudioHandler + streaming VAD (25 tests)
│   ├── test_pipeline.py            # RealTimeTranslator pipeline (35 tests)
│   ├── test_logging.py             # Logging setup + no-print enforcement (37 tests)
│   ├── test_recovery.py            # retry() + audio reconnect (27 tests)
│   ├── test_onnx.py                # ONNX Runtime routing + fallback (20 tests)
│   └── test_web.py                 # FastAPI routes + SSE broadcaster (36 tests)
│
├── scripts/
│   ├── run.sh                      # Main runner (venv, PYTHONPATH, conda isolation)
│   ├── run_translator.py           # Python entry point
│   ├── setup.sh                    # Automated setup
│   ├── test_setup.py               # Hardware verification
│   ├── download_models.py          # Pre-download ~5 GB of models
│   ├── create_config.py            # Interactive config generator
│   └── language_selector.py        # Standalone language picker
│
├── pyproject.toml                  # Package metadata + entry point
├── requirements.txt                # All dependencies (including optional)
├── pytest.ini                      # Test runner config
└── README.md
```

### Module Descriptions

| Module | Purpose |
|--------|---------|
| **src/main.py** | CLI argument parsing and application bootstrapping |
| **src/config.py** | Config dataclass: YAML load/save, nested section parsing, `from_yaml_or_default` |
| **src/models.py** | Hugging Face model loading for STT, translation, TTS; ONNX Runtime branch via Optimum |
| **src/audio_handler.py** | Microphone recording (`record_audio`), speaker playback, VAD-segmented streaming generator |
| **src/translator.py** | Orchestrates the full pipeline, bidirectional mode, history recording, retry wiring |
| **src/vad.py** | Silero VAD wrapper — filters silent chunks before STT |
| **src/lang_detect.py** | Whisper language detection + ISO 639-1 → NLLB code mapping (100 languages) |
| **src/lang_select.py** | Interactive language picker (used at startup when no lang is configured) |
| **src/tts_lang.py** | NLLB language code → `facebook/mms-tts-*` model name lookup |
| **src/history.py** | `ConversationHistory`: in-memory transcript, terminal display, JSONL file output |
| **src/validator.py** | Startup config validation — language codes, model IDs, sample rate constraints |
| **src/logging_setup.py** | Configures root logger (level, format, optional file handler) |
| **src/recovery.py** | `retry()` decorator with exponential backoff and per-exception filtering |
| **src/web.py** | FastAPI application: SSE transcript stream, REST API, embedded HTML UI |

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
   python scripts/create_config.py --interactive --output config/my_config.yaml
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
python scripts/create_config.py --list-languages
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

3. **ONNX Runtime Acceleration** (ARM CPU, no CUDA required):

   ONNX Runtime can give **2-4× faster CPU inference** on Raspberry Pi 5 by
   replacing PyTorch with an optimised ONNX execution engine.

   Install the optional dependency:
   ```bash
   pip install optimum[onnxruntime]
   ```

   Enable in `config/config.yaml`:
   ```yaml
   performance:
     use_onnx: true
   ```

   Then run as usual:
   ```bash
   ./scripts/run.sh --config config/config.yaml
   # or: raspberry-translator --config config/config.yaml
   ```

   **How it works**: on first run, models are exported to ONNX format (takes a
   few minutes). Subsequent runs reuse the cached ONNX graphs. If
   `optimum[onnxruntime]` is not installed, the system falls back to standard
   PyTorch and logs a warning.

   > **Note on Hailo-8L**: The Raspberry Pi AI Kit ships with a Hailo-8L NPU
   > offering up to 13 TOPS. Hailo inference requires the proprietary Hailo SDK
   > and HailoRT driver, which are not yet integrated here. ONNX Runtime on the
   > ARM CPU is the recommended path for most users. Hailo support can be added
   > by implementing a `HailoModelLoader` subclass that wraps the Hailo Python
   > API and returning its output in the same pipeline interface.

---

## 🛠 Troubleshooting

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
| `openai/whisper-tiny` | 39 M | ⚡⚡⚡ | ⭐⭐ | Testing, fastest response |
| `openai/whisper-base` | 74 M | ⚡⚡ | ⭐⭐⭐ | Balanced |
| `openai/whisper-small` | 244 M | ⚡ | ⭐⭐⭐⭐ | Production (default) |

Whisper also detects the spoken language automatically — set `auto_detect_source_lang: true` (default) to use this.

### Text-to-Speech Options
| Model pattern | Example | Notes |
|---------------|---------|-------|
| `facebook/mms-tts-{lang}` | `facebook/mms-tts-spa` | Auto-selected from target language (default) |
| Custom model | any HF model ID | Set `tts_auto_select: false` and specify `tts_model` |

### Bidirectional Conversation Mode

Two people speaking different languages can have a conversation with automatic direction switching:

```yaml
# config/config.yaml
languages:
  source: eng_Latn
  target: fra_Latn
  auto_detect_source_lang: true   # Whisper picks the direction each utterance
conversation:
  bidirectional: true
```

Or via CLI: `raspberry-translator -b`

### Streaming VAD Mode (Default)

Audio is segmented at natural speech boundaries using Silero VADIterator instead of fixed-duration recording chunks. Requires `sample_rate: 8000` or `16000` Hz.

```yaml
streaming:
  enabled: true          # Speech-boundary segmentation (recommended)
  min_silence_ms: 600    # Silence that ends an utterance
  max_duration_s: 30.0   # Hard cap per utterance
```

### Web Interface

```bash
# Launch the browser UI
raspberry-translator --web                         # http://localhost:7860
raspberry-translator --web --host 0.0.0.0 --port 8080

# REST API endpoints (also available without the UI)
GET  /api/status      # {"running": bool}
GET  /api/config      # Full config as JSON
GET  /api/history     # All transcript entries since server start
GET  /api/transcript  # Server-Sent Events stream of new entries
POST /api/start       # Start background translator
POST /api/stop        # Stop background translator
```

### Conversation History / Transcript

Each utterance is printed to the terminal and optionally saved:

```yaml
history:
  show: true
  save_path: "logs/session.jsonl"   # Each entry as a JSON line
  max_entries: 0                    # 0 = unlimited
```

### Logging

```bash
# Verbose per-chunk diagnostics
raspberry-translator --log-level DEBUG

# Save logs to file (in addition to stderr)
raspberry-translator --log-level INFO --log-file logs/translator.log
```

Log levels: `DEBUG` → `INFO` (default) → `WARNING` → `ERROR`

---

## 🗺️ Roadmap / TODOs

The following features are planned or pending implementation:

### Core Pipeline

- [x] **Voice Activity Detection (VAD)**: Integrate a VAD library (e.g., `silero-vad`) so audio chunks are only processed when actual speech is detected, avoiding wasted compute on silence.
- [x] **Dynamic TTS model selection**: Automatically switch the TTS model based on the configured target language (e.g., `facebook/mms-tts-spa` for Spanish). Currently the TTS model is fixed to English.
- [x] **Streaming / continuous STT**: Replace fixed-duration chunk recording with a streaming approach that segments audio at natural speech boundaries instead of arbitrary time intervals.
- [x] **Whisper language auto-detection integration**: Use Whisper's built-in language detection output to automatically align the NLLB source language code, eliminating the need for manual `source_lang` configuration.

### Features

- [x] **Bidirectional / conversation mode**: Add a push-to-talk or turn-taking mechanism so two speakers can have a back-and-forth conversation with real-time translation in both directions.
- [x] **Conversation history display**: Show a running transcript of detected speech and translations in the terminal during a session.
- [x] **Interactive language selection at startup**: Invoke `language_selector.py` interactively when no language is specified, rather than falling back to defaults silently.
- [x] **Web / GUI interface**: Build a simple web UI (e.g., Flask or FastAPI + HTML) or desktop GUI for configuration and live transcript display.

### Quality & Robustness

- [x] **Configuration validation**: Add validation for language codes (check against the NLLB supported list) and model identifiers at startup, with clear error messages.
- [x] **Proper logging**: Replace `print()` statements throughout the codebase with Python's `logging` module, supporting log levels and optional file output.
- [x] **Graceful error recovery**: Implement retry logic and clear user-facing messages when a model fails to load or an audio device becomes unavailable during a session.

### Developer Experience

- [x] **Unit test suite**: Add `pytest` tests for `Config`, `AudioHandler`, and the translation pipeline (with mocked models).
- [x] **Deduplicate root-level scripts**: `create_config.py`, `language_selector.py`, and `test_setup.py` exist both in the project root and in `scripts/`. Remove the duplicates from root.
- [x] **Package as installable module**: Add `pyproject.toml` / `setup.py` so the project can be installed with `pip install -e .` and `raspberry-translator` becomes an available CLI command.
- [x] **Raspberry Pi hardware acceleration**: Document and optionally support inference via Hailo-8L (available on RPi AI Kit) or ONNX Runtime for improved performance without CUDA.

---

## � Testing

The project ships with a full offline test suite (219 tests) that requires no audio hardware, GPU, or internet connection. All heavy dependencies are replaced with in-process stubs in `tests/conftest.py`.

```bash
source .venv/bin/activate

# Run all tests
python -m pytest

# Run a specific file with verbose output
python -m pytest tests/test_pipeline.py -v

# Run with short tracebacks
python -m pytest --tb=short
```

| Test file | Coverage area | Tests |
|-----------|---------------|------:|
| `test_config.py` | Config defaults, YAML round-trips, validation | 50 |
| `test_audio_handler.py` | Record, play, VAD streaming segments | 25 |
| `test_pipeline.py` | Translator pipeline (mock models) | 35 |
| `test_logging.py` | Logging setup, no-print enforcement | 37 |
| `test_recovery.py` | retry() logic, audio reconnect | 27 |
| `test_onnx.py` | ONNX routing + fallback | 20 |
| `test_web.py` | FastAPI routes, SSE broadcaster | 36 |

Install dev dependencies: `pip install -e ".[dev]"`

---

## �🤝 Contributing

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
