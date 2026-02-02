# Parakeet Model Support

The streaming transcription now supports NVIDIA's Parakeet TDT models in addition to OpenAI's Whisper models.

## What is Parakeet?

Parakeet is NVIDIA's high-performance multilingual ASR model that offers:
- **Superior accuracy** on many languages compared to Whisper
- **Automatic punctuation and capitalization**
- **Word-level timestamps**
- **25 European language support** with automatic language detection
- **600M parameter model** optimized for throughput

## Supported Languages

Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish, Russian, Ukrainian

## Installation

### 1. Install NeMo Toolkit

Parakeet models require the NVIDIA NeMo toolkit:

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Install NeMo with ASR support
pip install -r requirements_parakeet.txt
```

Or install directly:

```bash
pip install nemo_toolkit['asr']
```

### 2. Download Parakeet Model

Download the model (~600MB) before first use:

```bash
python scripts/download_parakeet_model.py
```

This script will:
- Check if NeMo is installed
- Download the model from Hugging Face with progress tracking
- Support resume if interrupted
- Verify the download completed successfully

**Note:** First-time download may take 2-10 minutes depending on your internet connection.

### 3. Update Configuration

Edit `config/streaming_config.yaml`:

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

**Or use the pre-configured file:**

```bash
# The project includes a ready-to-use Parakeet config
./scripts/run_streaming_transcribe.sh -c streaming_config_parakeet.yaml
```

### 4. Run

```bash
./scripts/run_streaming_transcribe.sh
```

Or with the Parakeet config:

```bash
./scripts/run_streaming_transcribe.sh -c streaming_config_parakeet.yaml
```

## Usage Examples

### Using Pre-configured Parakeet Config

A ready-to-use Parakeet configuration is included:

```bash
# Use the Parakeet config file (short form)
./scripts/run_streaming_transcribe.sh -c streaming_config_parakeet.yaml

# Use the Parakeet config file (long form)
./scripts/run_streaming_transcribe.sh --config config/streaming_config_parakeet.yaml
```

### Using Default Config with Parakeet

```bash
# Edit config/streaming_config.yaml to use Parakeet, then:
./scripts/run_streaming_transcribe.sh
```

### Override Model via Command Line

```bash
# Use Parakeet without editing config file
./scripts/run_streaming_transcribe.sh --model nvidia/parakeet-tdt-0.6b-v3

# Use Whisper instead
./scripts/run_streaming_transcribe.sh --model openai/whisper-small
```

### Spanish Transcription with Parakeet

```bash
./scripts/run_streaming_transcribe.sh --model nvidia/parakeet-tdt-0.6b-v3 --language es
```

## Performance Comparison

### Parakeet vs Whisper

**Parakeet advantages:**
- Better WER (Word Error Rate) on most European languages
- Automatic punctuation and capitalization
- Better handling of multilingual content
- Optimized for throughput

**Whisper advantages:**
- Smaller model options (tiny, base)
- No additional dependencies
- More language coverage (100+ languages)
- Better for very low-resource devices

### Model Sizes

| Model | Parameters | Languages | Speed |
|-------|-----------|-----------|-------|
| whisper-tiny | 39M | 100+ | Fastest |
| whisper-small | 244M | 100+ | Fast |
| parakeet-tdt-0.6b-v3 | 600M | 25 | Medium |
| whisper-medium | 769M | 100+ | Slow |

## Example Performance (WER %)

Based on FLEURS dataset:

| Language | Parakeet | Whisper Small |
|----------|----------|---------------|
| Spanish  | 3.45%    | ~5-6% |
| French   | 5.15%    | ~6-7% |
| German   | 5.04%    | ~6-7% |
| English  | 4.85%    | ~5-6% |
| Italian  | 3.00%    | ~4-5% |

## Switching Between Models

You can easily switch between Whisper and Parakeet:

### In Configuration File

```yaml
# For Whisper
model:
  name: "openai/whisper-small"

# For Parakeet  
model:
  name: "nvidia/parakeet-tdt-0.6b-v3"
```

### Via Command Line

```bash
# Whisper
./scripts/run_streaming_transcribe.sh --model openai/whisper-small --language es

# Parakeet
./scripts/run_streaming_transcribe.sh --model nvidia/parakeet-tdt-0.6b-v3 --language es
```

## Troubleshooting

### NeMo Not Installed

**Error:** `ImportError: No module named 'nemo'`

**Solution:**
```bash
pip install nemo_toolkit['asr']
```

### Model Download Issues

Parakeet models are ~2.4GB. First run will download the model:

```bash
# The model will be cached at ~/.cache/huggingface/
# Subsequent runs will use the cached version
```

### Memory Issues

Parakeet requires at least 2GB RAM. For Raspberry Pi:
- Use swap space if needed
- Close other applications
- Consider using whisper-tiny or whisper-small instead

## Technical Details

### Model Architecture

- **Type:** FastConformer-TDT (Token and Duration Transducer)
- **Framework:** NVIDIA NeMo
- **Input:** 16kHz mono audio
- **Output:** Text with punctuation and capitalization
- **License:** CC-BY-4.0

### Implementation Details

The transcription program:
1. Detects model type from model name
2. Uses NeMo API for Parakeet models
3. Uses Transformers pipeline for Whisper models
4. Handles both audio array and file-based transcription
5. Maintains the same interface for both model types

## References

- [Parakeet Model Card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [NeMo Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html)
- [Technical Paper](https://arxiv.org/abs/2509.14128)
