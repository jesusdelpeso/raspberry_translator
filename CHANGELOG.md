# Changelog

All notable changes to the Raspberry Pi Real-time Audio Translator project will be documented in this file.

## [Unreleased] - 2026-02-02

### Fixed - Audio Streaming (PortAudio/ALSA)
- **Critical fix for audio capture** (`streaming_transcribe.py`)
  - Fixed PortAudio threading timeout errors on Linux systems
  - Changed from callback-based to blocking read mode for audio capture
  - Resolved "Wait timed out [PaErrorCode -9987]" error
  - Added retry logic (3 attempts) for audio stream initialization
  - Improved error handling with detailed troubleshooting messages
  - Optimized block size (reduced from 48,000 to 1,024 samples)
  - Added buffer overflow detection and reporting
  - Audio streaming now works reliably on all tested Linux systems

- **Documentation**
  - `AUDIO_STREAMING_FIXES.md` - Comprehensive technical documentation
  - Details root cause, solution, and testing results
  - Usage recommendations for production deployment
  - Technical comparison of callback vs blocking read modes

### Added - Config File Selection
- **Enhanced shell script** (`run_streaming_transcribe.sh`)
  - Added `-c` / `--config` parameter to specify configuration file
  - Automatic path resolution from config/ directory
  - Built-in help message with usage examples
  - Config file existence validation with helpful error messages
  - Lists available configs when specified file not found
  - Backward compatible - still uses default config when not specified

### Added - Parakeet Model Setup Tools
- **Download helper script** (`scripts/download_parakeet_model.py`)
  - Automated download of Parakeet model from Hugging Face
  - Progress tracking and resume support for ~600MB download
  - Validates NeMo installation before downloading
  - Shows download status and file location
  - Clear error messages and troubleshooting guidance

### Fixed - Parakeet Model Setup
- **Resolved dependency issues**
  - Installed nemo_toolkit[asr] for Parakeet model support
  - Documented model download requirement (~600MB)
  - Created step-by-step setup guide in PARAKEET_FIX.md
  - Added verification steps for installation

### Added - Parakeet Model Support
- **Multi-model ASR support** (`streaming_transcribe.py`)
  - Added support for NVIDIA Parakeet TDT models alongside Whisper
  - Automatic model type detection from model name
  - Native NeMo integration for Parakeet models
  - Maintains unified interface for both Whisper and Parakeet
  - Superior accuracy with `nvidia/parakeet-tdt-0.6b-v3` (600M params, 25 languages)
  - Automatic punctuation and capitalization with Parakeet
  - Word-level timestamps support (Parakeet)

- **Documentation**
  - `PARAKEET_SUPPORT.md` - Complete guide for Parakeet models
  - `requirements_parakeet.txt` - Optional dependencies for Parakeet
  - Updated configuration examples with Parakeet options
  - Performance comparison between Whisper and Parakeet
  - Installation and usage instructions

- **Configuration enhancements**
  - Added Parakeet model to config examples
  - Updated model selection documentation
  - Added tips for choosing between Whisper and Parakeet

### Fixed - Streaming Transcription
- **Fixed crash on interrupt** (`streaming_transcribe.py`)
  - Resolved "terminate called without an active exception" error when pressing Ctrl+C
  - Changed processing thread to daemon thread for proper cleanup
  - Implemented context manager for audio stream resource management
  - Added proper exception handling for KeyboardInterrupt and general exceptions
  - Added timeout to thread join to prevent hanging
  - Program now exits gracefully without crashes

## [2026-02-01]

### Added - Streaming Transcription Feature
- **New streaming transcription application** (`scripts/streaming_transcribe.py`)
  - Real-time audio transcription with sentence-by-sentence processing
  - Voice Activity Detection (VAD) for intelligent audio capture
  - Dual sentence detection: punctuation-based and silence-based
  - Configurable parameters via command-line arguments
  - Shell wrapper script (`scripts/run_streaming_transcribe.sh`)

- **Language parameter support**
  - `--language` parameter for specifying transcription language
  - Supports 14+ common languages (en, es, fr, de, it, pt, nl, pl, ru, zh, ja, ko, ar, hi)
  - Auto-detect mode when language not specified
  - Improves accuracy and speed for known languages

- **Comprehensive documentation**
  - `STREAMING_TRANSCRIPTION.md` - Complete user guide
  - `STREAMING_TRANSCRIPTION_FIXES.md` - Testing and troubleshooting documentation
  - Updated README.md with streaming transcription section
  - Usage examples and parameter tuning guides

### Fixed
- Python command compatibility in `run_streaming_transcribe.sh` (changed from `python` to `python3`)
- Added warning suppression for cleaner output during transcription
- Set environment variables for better transformers library behavior

### Tested
- Audio device detection on Linux systems
- Streaming transcription with whisper-tiny model
- Spanish language transcription (`--language es`)
- Voice Activity Detection functionality
- Sentence boundary detection
- Clean shutdown with Ctrl+C

## Initial Release

### Features
- Real-time translation pipeline (Speech-to-Text → Translation → Text-to-Speech)
- Support for 200+ languages via NLLB-200 translation model
- OpenAI Whisper for speech recognition
- Configurable YAML-based settings
- Raspberry Pi 5 optimized
- Multiple helper scripts for setup and configuration
- Comprehensive documentation and troubleshooting guides

### Models
- **STT**: openai/whisper-small (default)
- **Translation**: facebook/nllb-200-distilled-1.3B
- **TTS**: facebook/mms-tts-eng (default)

### Scripts
- `run_translator.py` - Main translation application
- `run.sh` - Convenience shell runner
- `setup.sh` - Automated setup
- `test_setup.py` - System verification
- `download_models.py` - Pre-download AI models
- `create_config.py` - Configuration generator
- `language_selector.py` - Interactive language picker
