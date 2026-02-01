# Changelog

All notable changes to the Raspberry Pi Real-time Audio Translator project will be documented in this file.

## [Unreleased] - 2026-02-02

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
