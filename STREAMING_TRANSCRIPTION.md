# Streaming Audio Transcription

A real-time audio transcription tool that captures audio from your microphone and transcribes it sentence by sentence using OpenAI's Whisper model.

## Features

- **Real-time streaming transcription**: Captures and processes audio continuously
- **Sentence-by-sentence processing**: Detects sentence boundaries and completes each sentence before starting the next
- **Voice Activity Detection (VAD)**: Automatically detects speech vs. silence
- **Configurable parameters**: Adjust chunk duration, VAD threshold, and silence detection
- **GPU support**: Optional GPU acceleration for faster transcription

## How It Works

1. **Audio Capture**: Continuously captures audio from the microphone in small chunks (default: 3 seconds)
2. **Voice Detection**: Uses amplitude-based VAD to detect when you're speaking
3. **Buffering**: Accumulates audio chunks while speech is detected
4. **Sentence Detection**: 
   - Periodically checks for sentence-ending punctuation (. ! ?)
   - Detects silence periods to mark sentence boundaries
5. **Transcription**: When a sentence is complete, transcribes the entire sentence buffer
6. **Reset**: Clears the buffer and starts capturing the next sentence

## Usage

### Basic Usage

```bash
# Run with default settings
./scripts/run_streaming_transcribe.sh

# Or run directly with Python
python scripts/streaming_transcribe.py
```

### With Options

```bash
# Use GPU acceleration
./scripts/run_streaming_transcribe.sh --gpu

# Use a different Whisper model
./scripts/run_streaming_transcribe.sh --model openai/whisper-base

# Specify language for better accuracy (e.g., Spanish)
./scripts/run_streaming_transcribe.sh --language es

# English transcription
./scripts/run_streaming_transcribe.sh --language en

# Adjust VAD threshold (lower = more sensitive)
./scripts/run_streaming_transcribe.sh --vad-threshold 0.01

# Adjust silence duration for sentence end detection
./scripts/run_streaming_transcribe.sh --silence-duration 2.0

# Adjust audio chunk duration
./scripts/run_streaming_transcribe.sh --chunk-duration 2.0
```

### List Available Audio Devices

```bash
./scripts/run_streaming_transcribe.sh --list-devices
```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `openai/whisper-small` | Whisper model to use |
| `--sample-rate` | `16000` | Audio sample rate in Hz |
| `--chunk-duration` | `3.0` | Duration of audio chunks in seconds |
| `--vad-threshold` | `0.02` | Voice activity detection threshold (0.0-1.0) |
| `--silence-duration` | `1.5` | Silence duration to mark end of sentence (seconds) |
| `--language` | `None` | Language code (e.g., 'en', 'es', 'fr'). Auto-detect if not specified. |
| `--gpu` | `False` | Use GPU if available |
| `--list-devices` | - | List available audio devices and exit |

## Available Whisper Models

From smallest/fastest to largest/most accurate:

- `openai/whisper-tiny` - Fastest, lowest accuracy
- `openai/whisper-base` - Good balance for testing
- `openai/whisper-small` - **Default**, good accuracy
- `openai/whisper-medium` - Better accuracy, slower
- `openai/whisper-large-v3` - Best accuracy, requires more resources

## Parameter Tuning

### Language (`--language`)

Specifying the language improves transcription accuracy and speed:

- **Auto-detect (default)**: Whisper automatically detects the language from the audio
- **Specified language**: Forces transcription in a specific language for better accuracy

**Common language codes:**
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `nl` - Dutch
- `pl` - Polish
- `ru` - Russian
- `zh` - Chinese
- `ja` - Japanese
- `ko` - Korean
- `ar` - Arabic
- `hi` - Hindi

**When to use:**
- Use auto-detect for multilingual conversations or when unsure
- Specify language when you know what language will be spoken for better accuracy

### VAD Threshold (`--vad-threshold`)

- **Lower values (0.01)**: More sensitive, captures quieter speech but may pick up background noise
- **Higher values (0.05)**: Less sensitive, only captures louder speech
- **Default (0.02)**: Good balance for normal speaking volume

### Silence Duration (`--silence-duration`)

- **Shorter (1.0s)**: Faster sentence detection, but may cut off long pauses within sentences
- **Longer (2.5s)**: More tolerant of pauses, but slower to detect sentence ends
- **Default (1.5s)**: Good balance for natural speech patterns

### Chunk Duration (`--chunk-duration`)

- **Shorter (1.0-2.0s)**: More responsive, but more frequent processing
- **Longer (4.0-5.0s)**: Less CPU usage, but slower response
- **Default (3.0s)**: Good balance for real-time performance

## Example Sessions

### Basic Transcription
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
^C
Stopping transcription...

Total sentences transcribed: 2
```

### With GPU and Custom Model
```bash
$ ./scripts/run_streaming_transcribe.sh --gpu --model openai/whisper-base --vad-threshold 0.015

Using device: cuda
Loading model: openai/whisper-base...
Model loaded successfully!
...
```

### Spanish Transcription
```bash
$ ./scripts/run_streaming_transcribe.sh --language es

Using device: cpu
Loading model: openai/whisper-small...
Model loaded successfully! (language: es)

================================================================================
STREAMING AUDIO TRANSCRIPTION
================================================================================
Sample rate: 16000 Hz
Chunk duration: 3.0s
VAD threshold: 0.02
Silence duration for sentence end: 1.5s
Language: es
================================================================================

Processing audio... Speak into the microphone.
Press Ctrl+C to stop.

--------------------------------------------------------------------------------

[Sentence 1]: Hola, ¿cómo estás hoy?
--------------------------------------------------------------------------------
```

## Troubleshooting

### No Audio Input

**Problem**: No audio is being captured

**Solutions**:
- List available devices: `./scripts/run_streaming_transcribe.sh --list-devices`
- Check microphone permissions in your OS
- Test microphone with: `arecord -d 3 test.wav` (Linux)

### Low Transcription Accuracy

**Solutions**:
- Specify the language explicitly: `--language en` (or your language code)
- Use a larger model: `--model openai/whisper-medium`
- Speak more clearly and at a consistent volume
- Reduce background noise
- Adjust VAD threshold: `--vad-threshold 0.015`

### Sentences Cut Off Too Early

**Solutions**:
- Increase silence duration: `--silence-duration 2.5`
- Speak with fewer pauses between words

### Sentences Not Ending

**Solutions**:
- Decrease silence duration: `--silence-duration 1.0`
- Speak more clearly with distinct sentence endings
- Use more pronounced pauses between sentences

### High CPU/Memory Usage

**Solutions**:
- Use a smaller model: `--model openai/whisper-base` or `--model openai/whisper-tiny`
- Increase chunk duration: `--chunk-duration 5.0`
- Use GPU acceleration: `--gpu`

## Technical Details

### Voice Activity Detection (VAD)

The script uses a simple but effective amplitude-based VAD:
- Calculates RMS (Root Mean Square) of each audio chunk
- Compares RMS to threshold to detect speech
- Tracks consecutive silence chunks to determine sentence boundaries

### Sentence Detection Strategy

Two methods for detecting sentence ends:

1. **Punctuation-based**: Checks transcribed text for sentence-ending punctuation (. ! ?)
2. **Silence-based**: Detects silence periods lasting longer than the configured duration

This dual approach ensures sentences are properly segmented even when punctuation is ambiguous.

### Audio Processing Pipeline

```
Microphone → Audio Chunks → Voice Detection → Buffer Accumulation
                                                       ↓
                                              Sentence Detection
                                                       ↓
                                              Transcription (Whisper)
                                                       ↓
                                              Display & Clear Buffer
```

## Performance Considerations

- **Model size**: Larger models are more accurate but slower
- **Chunk duration**: Smaller chunks = more responsive but more CPU
- **GPU acceleration**: Significantly faster for medium/large models
- **Buffer size**: Accumulates audio in memory until sentence completion

## Differences from Translation App

This streaming transcription app differs from the main translation application:

- **Continuous operation**: Runs indefinitely until stopped (vs. fixed-duration chunks)
- **Sentence-aware**: Detects and processes complete sentences
- **Streaming mode**: Real-time processing with immediate feedback
- **Transcription only**: No translation or text-to-speech
- **Lower latency**: Optimized for quick feedback

## Integration Ideas

This script can be used as a component for:
- Real-time captioning systems
- Voice command interfaces
- Meeting transcription tools
- Language learning applications
- Accessibility tools

## Requirements

All dependencies are already included in the main project's `requirements.txt`:
- torch
- transformers
- sounddevice
- numpy

## License

Same as the main raspberry_translator project.
