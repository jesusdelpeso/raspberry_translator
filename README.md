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
- 🌍 **200+ Languages**: Supports translation between over 200 languages
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
| **Translation** | `facebook/nllb-200-distilled-1.3B` | Supports 200+ languages with good quality |
| **Text-to-Speech** | `facebook/mms-tts-eng` | Lightweight, suitable for edge devices |

*All models are customizable via configuration file*

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Language Codes](#-language-codes)
- [Performance Tips](#-performance-tips)
- [Troubleshooting](#-troubleshooting)
- [Advanced Usage](#-advanced-usage)
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
```

### 3. Configure the Application

Copy the example configuration and customize it:

```bash
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml with your preferred settings
```

### 4. Download Models (Optional)

Models will be automatically downloaded on first run, but you can pre-download them:

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
│   ├── setup.sh                   # Automated setup script
│   ├── test_setup.py              # System verification tests
│   ├── download_models.py         # Pre-download AI models
│   ├── create_config.py           # Configuration generator
│   └── language_selector.py       # Interactive language picker
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
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
