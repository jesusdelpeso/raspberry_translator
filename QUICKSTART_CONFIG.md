# Quick Start Guide - Streaming Transcription with Config File

## What Changed?

The streaming transcription script now uses a configuration file by default, making it much easier to:
- Run the script without remembering all parameters
- Share configuration settings
- Switch between different setups quickly

## Quick Start

### 1. Edit the Configuration File

Edit `config/streaming_config.yaml` to set your preferences:

```yaml
model:
  name: "openai/whisper-small"  # Change model size here
  use_gpu: false                 # Set to true if you have CUDA

audio:
  sample_rate: 16000
  chunk_duration: 3.0            # Adjust responsiveness
  vad_threshold: 0.02            # Adjust sensitivity
  silence_duration: 1.5          # Adjust sentence breaks

language:
  code: "es"                     # Change to your language (en, fr, de, etc.)
```

### 2. Run the Script

Simply run without any arguments:

```bash
./scripts/run_streaming_transcribe.sh
```

That's it! The script will use your configuration file automatically.

## Common Scenarios

### Running with Default Config

```bash
# Uses config/streaming_config.yaml
./scripts/run_streaming_transcribe.sh
```

### Using a Specific Config File

```bash
# Use Parakeet config (short form)
./scripts/run_streaming_transcribe.sh -c streaming_config_parakeet.yaml

# Use Parakeet config (long form)
./scripts/run_streaming_transcribe.sh --config streaming_config_parakeet.yaml

# With full path
./scripts/run_streaming_transcribe.sh --config config/streaming_config_parakeet.yaml
```

### Override Just One Setting

```bash
# Use English instead of the language in config file
./scripts/run_streaming_transcribe.sh --language en

# Use a different model
./scripts/run_streaming_transcribe.sh --model openai/whisper-tiny

# Combine config file with overrides
./scripts/run_streaming_transcribe.sh -c streaming_config_parakeet.yaml --language en
```

### List Available Audio Devices

```bash
./scripts/run_streaming_transcribe.sh --list-devices
```

## Configuration Profiles

You can create multiple configuration files for different scenarios:

### English Fast Transcription

Save as `config/streaming_en_fast.yaml`:
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
  code: "en"
```

Run with:
```bash
# With full path
./scripts/run_streaming_transcribe.sh --config config/streaming_en_fast.yaml

# Or just the filename (auto-resolves from config/ directory)
./scripts/run_streaming_transcribe.sh -c streaming_en_fast.yaml
```

### Spanish High Accuracy

Save as `config/streaming_es_accurate.yaml`:
```yaml
model:
  name: "openai/whisper-small"
  use_gpu: false
audio:
  sample_rate: 16000
  chunk_duration: 4.0
  vad_threshold: 0.01
  silence_duration: 2.0
language:
  code: "es"
```

Run with:
```bash
./scripts/run_streaming_transcribe.sh --config config/streaming_es_accurate.yaml
```

## Benefits

✓ **Easier to use** - No need to remember all command line options
✓ **Repeatable** - Same settings every time
✓ **Shareable** - Share your config file with others
✓ **Flexible** - Override any setting when needed
✓ **Multiple profiles** - Create configs for different scenarios

## Tips

1. **Start with the default config** and adjust based on your needs
2. **Test different models** to find the best balance of speed/accuracy for your hardware
3. **Adjust VAD threshold** based on your environment (quieter = lower value)
4. **Create profiles** for different languages or use cases
5. **Use command line overrides** for quick tests without editing the config file

## Troubleshooting

**Config file not found?**
- Make sure `config/streaming_config.yaml` exists
- Or specify a different config file with `--config`

**Settings not being used?**
- Check the console output - it shows which config file is loaded
- Remember: command line arguments override config file settings

**Need old behavior?**
- You can still use all command line arguments like before
- Just specify all parameters on command line (config file is optional)
