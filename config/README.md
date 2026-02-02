# Streaming Configuration

This directory contains configuration files for the streaming transcription application.

## Configuration Files

### streaming_config.yaml

Main configuration file for the streaming transcription feature. This file controls all aspects of audio transcription behavior.

## Usage

### Using Config File Only

Simply run the script without arguments to use settings from the config file:

```bash
./scripts/run_streaming_transcribe.sh
```

### Overriding Config Settings

You can override any config file setting using command line arguments:

```bash
# Override just the language
./scripts/run_streaming_transcribe.sh --language en

# Override model and chunk duration
./scripts/run_streaming_transcribe.sh --model openai/whisper-tiny --chunk-duration 2.0

# Use GPU (overrides config file)
./scripts/run_streaming_transcribe.sh --gpu
```

### Using a Different Config File

```bash
# With full path
./scripts/run_streaming_transcribe.sh --config /path/to/my_config.yaml

# With relative path
./scripts/run_streaming_transcribe.sh --config config/streaming_config_parakeet.yaml

# Just filename (auto-resolves from config/ directory)
./scripts/run_streaming_transcribe.sh -c streaming_config_parakeet.yaml

# Short form also works
./scripts/run_streaming_transcribe.sh -c my_config.yaml
```

**Note:** If you specify just a filename (no path), the script automatically looks in the `config/` directory.

## Configuration Options

### Model Settings

- **name**: ASR model to use
  
  **Whisper Models (OpenAI):**
  - `openai/whisper-tiny` - Fastest, least accurate (~39M params)
  - `openai/whisper-base` - Fast, decent accuracy (~74M params)
  - `openai/whisper-small` - Good balance (recommended, ~244M params)
  - `openai/whisper-medium` - Slower, more accurate (~769M params)
  - `openai/whisper-large-v2` - Slowest, most accurate (~1.5B params)
  
  **Parakeet Models (NVIDIA NeMo):**
  - `nvidia/parakeet-tdt-0.6b-v3` - High accuracy multilingual (600M params, 25 European languages)
    - Supports: English, Spanish, French, German, Italian, Portuguese, Dutch, Polish, Russian, and 16 more
    - Features: Automatic punctuation, capitalization, word-level timestamps
    - Better performance than Whisper on many languages
    - Requires: `nemo_toolkit['asr']` package

- **use_gpu**: Whether to use GPU acceleration (requires CUDA)
  - `true` or `false`

### Audio Settings

- **sample_rate**: Audio sample rate in Hz
  - Default: `16000` (standard for Whisper)

- **chunk_duration**: Duration of audio chunks to process (seconds)
  - Smaller values = more responsive
  - Larger values = better context
  - Default: `3.0`

- **vad_threshold**: Voice Activity Detection threshold (0.0 to 1.0)
  - Lower = more sensitive to quiet sounds
  - Higher = only detects louder sounds
  - Default: `0.02`

- **silence_duration**: Silence duration to end a sentence (seconds)
  - Shorter = more frequent sentence breaks
  - Longer = longer sentences before breaking
  - Default: `1.5`

### Language Settings

- **code**: Language code for transcription
  - Use specific language code for better accuracy: `en`, `es`, `fr`, `de`, etc.
  - Use `auto` for automatic language detection
  - Default: `es` (Spanish)

## Examples

### Configuration for English Transcription

```yaml
model:
  name: "openai/whisper-small"
  use_gpu: false

audio:
  sample_rate: 16000
  chunk_duration: 3.0
  vad_threshold: 0.02
  silence_duration: 1.5

language:
  code: "en"
```

### Configuration for Fast, Responsive Transcription

```yaml
model:
  name: "openai/whisper-tiny"
  use_gpu: false

audio:
  sample_rate: 16000
  chunk_duration: 2.0
  vad_threshold: 0.03
  silence_duration: 1.0

language:
  code: "auto"
```

### Configuration for High Accuracy (Slower)

```yaml
model:
  name: "openai/whisper-medium"
  use_gpu: true

audio:
  sample_rate: 16000
  chunk_duration: 4.0
  vad_threshold: 0.01
  silence_duration: 2.0

language:
  code: "en"
```

### Configuration for Parakeet Model (Best Multilingual Accuracy)

```yaml
model:
  name: "nvidia/parakeet-tdt-0.6b-v3"
  use_gpu: false

audio:
  sample_rate: 16000
  chunk_duration: 3.0
  vad_threshold: 0.02
  silence_duration: 1.5

language:
  code: "auto"  # Parakeet auto-detects language
```

## Tips

1. **For Raspberry Pi**: Use `whisper-tiny` or `whisper-small` for best performance
2. **For best accuracy**: Use `nvidia/parakeet-tdt-0.6b-v3` (requires NeMo toolkit)
3. **For European languages**: Parakeet often outperforms Whisper
4. **For quiet environments**: Lower the `vad_threshold` (e.g., 0.01)
5. **For noisy environments**: Raise the `vad_threshold` (e.g., 0.05)
6. **For short phrases**: Reduce `silence_duration` (e.g., 1.0)
7. **For long sentences**: Increase `silence_duration` (e.g., 2.5)
8. **Known language**: Always specify language code for Whisper (Parakeet auto-detects)

## Installing NeMo for Parakeet Models

To use Parakeet models, install the NeMo toolkit:

```bash
pip install nemo_toolkit['asr']
```
