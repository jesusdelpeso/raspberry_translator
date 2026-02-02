#!/usr/bin/env python3
"""
Streaming audio transcription with sentence-by-sentence processing.

This script captures audio from the microphone and transcribes it in real-time
using a streaming approach. It detects sentence boundaries and completes each
sentence transcription before starting a new one.
"""

import sys
import os

# Set environment variables before importing transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import queue
import threading
import time
import re
import warnings
from typing import Optional
import yaml
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import logging

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*custom logits processor.*")
warnings.filterwarnings("ignore", message=".*chunk_length_s.*experimental.*")
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*deprecated.*")
warnings.filterwarnings("ignore", message=".*language detection.*")
warnings.filterwarnings("ignore", message=".*generation_config.*deprecated.*")

# Set transformers logging to error only
logging.getLogger("transformers").setLevel(logging.ERROR)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config


class StreamingTranscriber:
    """Handles streaming audio transcription with sentence detection"""

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        sample_rate: int = 16000,
        chunk_duration: float = 3.0,
        vad_threshold: float = 0.02,
        silence_duration: float = 1.5,
        use_gpu: bool = False,
        language: Optional[str] = None,
    ):
        """
        Initialize streaming transcriber
        
        Args:
            model_name: Whisper model to use
            sample_rate: Audio sample rate in Hz
            chunk_duration: Duration of audio chunks to process (seconds)
            vad_threshold: Threshold for voice activity detection
            silence_duration: Duration of silence to consider end of sentence (seconds)
            use_gpu: Whether to use GPU if available
            language: Language code for transcription (e.g., 'en', 'es', 'fr'). If None, auto-detect.
        """
        self.sample_rate = sample_rate
        self.language = language
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.vad_threshold = vad_threshold
        self.silence_duration = silence_duration
        self.silence_chunks_needed = int(silence_duration / chunk_duration)
        
        # Audio buffers
        self.audio_queue = queue.Queue()
        self.current_sentence_buffer = []
        self.silence_counter = 0
        self.is_recording = False
        
        # Setup device
        use_cuda = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.device_index = 0 if self.device.type == "cuda" else -1
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        print(f"Using device: {self.device.type}")
        print(f"Loading model: {model_name}...")
        
        # Detect model type
        self.model_type = "whisper" if "whisper" in model_name.lower() else "parakeet" if "parakeet" in model_name.lower() else "whisper"
        
        if self.model_type == "parakeet":
            # Load Parakeet model using NeMo
            try:
                import nemo.collections.asr as nemo_asr
                self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
                if use_cuda:
                    self.model = self.model.cuda()
                self.pipe = None  # Parakeet doesn't use pipeline
                print(f"Parakeet model loaded successfully!\n")
            except ImportError:
                print("Error: nemo_toolkit is required for Parakeet models.")
                print("Install with: pip install nemo_toolkit['asr']")
                raise
        else:
            # Load Whisper model
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.to(self.device)
            
            processor = AutoProcessor.from_pretrained(model_name)
            
            # Build pipeline kwargs
            pipe_kwargs = {
                "task": "automatic-speech-recognition",
                "model": model,
                "tokenizer": processor.tokenizer,
                "feature_extractor": processor.feature_extractor,
                "max_new_tokens": 128,
                "chunk_length_s": 30,
                "batch_size": 16,
                "return_timestamps": False,
                "torch_dtype": self.dtype,
                "device": self.device_index,
            }
            
            # Add generate_kwargs for language if specified
            if self.language:
                pipe_kwargs["generate_kwargs"] = {"language": self.language}
            
            self.pipe = pipeline(**pipe_kwargs)
            self.model = None
            
            language_msg = f" (language: {self.language})" if self.language else " (auto-detect language)"
            print(f"Whisper model loaded successfully!{language_msg}\n")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Status: {status}", file=sys.stderr)
        
        # Add audio chunk to queue
        self.audio_queue.put(indata.copy())

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Simple voice activity detection based on audio amplitude
        
        Args:
            audio_chunk: Audio data as numpy array
            
        Returns:
            True if speech is detected, False otherwise
        """
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        return rms > self.vad_threshold

    def has_sentence_ending(self, text: str) -> bool:
        """
        Check if text contains a sentence ending marker
        
        Args:
            text: Transcribed text
            
        Returns:
            True if sentence ending is detected
        """
        if not text:
            return False
        
        # Check for sentence-ending punctuation
        sentence_endings = re.compile(r'[.!?]\s*$')
        return bool(sentence_endings.search(text.strip()))

    def transcribe_buffer(self, audio_buffer: list) -> Optional[str]:
        """
        Transcribe accumulated audio buffer
        
        Args:
            audio_buffer: List of audio chunks
            
        Returns:
            Transcribed text or None if buffer is empty
        """
        if not audio_buffer:
            return None
        
        # Concatenate all chunks
        audio_array = np.concatenate(audio_buffer, axis=0)
        
        # Convert to 1D if needed
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()
        
        # Skip if audio is too quiet
        if np.max(np.abs(audio_array)) < self.vad_threshold:
            return None
        
        try:
            if self.model_type == "parakeet":
                # Parakeet transcription
                # Save temporary audio file (Parakeet expects file paths)
                import tempfile
                import soundfile as sf
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    sf.write(tmp_path, audio_array, self.sample_rate)
                
                try:
                    result = self.model.transcribe([tmp_path])
                    # Result is a list of strings for Parakeet, not objects
                    if result and len(result) > 0:
                        text = result[0] if isinstance(result[0], str) else str(result[0])
                        text = text.strip()
                    else:
                        text = None
                finally:
                    # Clean up temp file
                    import os
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                
                return text if text else None
            else:
                # Whisper transcription
                result = self.pipe(audio_array)
                text = result["text"].strip()
                return text if text else None
        except Exception as e:
            import traceback
            print(f"Transcription error: {e}", file=sys.stderr)
            print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
            return None

    def process_audio_stream(self):
        """Process audio from queue and transcribe sentence by sentence"""
        print("Processing audio... Speak into the microphone.")
        print("Press Ctrl+C to stop.\n")
        print("-" * 80)
        
        sentence_count = 0
        
        try:
            while self.is_recording:
                try:
                    # Get audio chunk from queue (with timeout)
                    audio_chunk = self.audio_queue.get(timeout=0.5)
                    
                    # Check for speech activity
                    has_speech = self.is_speech(audio_chunk)
                    
                    if has_speech:
                        # Add to current sentence buffer
                        self.current_sentence_buffer.append(audio_chunk)
                        self.silence_counter = 0
                        
                        # Periodically check for sentence endings
                        if len(self.current_sentence_buffer) >= 3:  # At least ~9 seconds
                            partial_text = self.transcribe_buffer(self.current_sentence_buffer)
                            
                            if partial_text and self.has_sentence_ending(partial_text):
                                # Complete sentence detected
                                sentence_count += 1
                                print(f"\n[Sentence {sentence_count}]: {partial_text}")
                                print("-" * 80)
                                
                                # Clear buffer for next sentence
                                self.current_sentence_buffer = []
                                self.silence_counter = 0
                    
                    else:
                        # Silence detected
                        if self.current_sentence_buffer:
                            self.silence_counter += 1
                            
                            # If enough silence and we have content, finish the sentence
                            if self.silence_counter >= self.silence_chunks_needed:
                                final_text = self.transcribe_buffer(self.current_sentence_buffer)
                                
                                if final_text:
                                    sentence_count += 1
                                    print(f"\n[Sentence {sentence_count}]: {final_text}")
                                    print("-" * 80)
                                
                                # Clear buffer for next sentence
                                self.current_sentence_buffer = []
                                self.silence_counter = 0
                
                except queue.Empty:
                    continue
                
        except KeyboardInterrupt:
            print("\n\nStopping transcription...")
        
        # Process any remaining audio
        if self.current_sentence_buffer:
            final_text = self.transcribe_buffer(self.current_sentence_buffer)
            if final_text:
                sentence_count += 1
                print(f"\n[Sentence {sentence_count}]: {final_text}")
                print("-" * 80)
        
        print(f"\nTotal sentences transcribed: {sentence_count}")

    def start_streaming(self):
        """Start streaming audio capture and transcription"""
        print("=" * 80)
        print("STREAMING AUDIO TRANSCRIPTION")
        print("=" * 80)
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Chunk duration: {self.chunk_duration}s")
        print(f"VAD threshold: {self.vad_threshold}")
        print(f"Silence duration for sentence end: {self.silence_duration}s")
        language_display = self.language if self.language else "auto-detect"
        print(f"Language: {language_display}")
        print("=" * 80)
        print()
        
        self.is_recording = True
        
        # Start audio processing thread
        processing_thread = threading.Thread(target=self.process_audio_stream, daemon=True)
        processing_thread.start()
        
        # Start audio stream with polling instead of callback to avoid threading issues
        stream = None
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Use smaller blocksize to avoid timeout issues
                blocksize = min(self.chunk_samples, 1024)
                
                # Try using blocking read instead of callback
                print("Attempting to open audio stream (blocking mode)...", file=sys.stderr)
                stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype=np.float32,
                    blocksize=blocksize,
                )
                
                # Start the stream
                stream.start()
                print(f"Audio stream started successfully!")
                
                try:
                    # Read audio in blocking mode and put in queue
                    while self.is_recording:
                        # Read audio data
                        audio_data, overflowed = stream.read(blocksize)
                        if overflowed:
                            print("Audio buffer overflow detected", file=sys.stderr)
                        
                        # Add to queue for processing
                        self.audio_queue.put(audio_data)
                        
                except KeyboardInterrupt:
                    print("\n\nReceived interrupt signal...")
                finally:
                    # Stop and close the stream safely
                    try:
                        if hasattr(stream, 'active') and stream.active:
                            stream.stop()
                    except Exception as stop_error:
                        print(f"Warning: Error stopping stream: {stop_error}", file=sys.stderr)
                    
                    try:
                        stream.close()
                    except Exception as close_error:
                        print(f"Warning: Error closing stream: {close_error}", file=sys.stderr)
                
                # If we got here, the stream worked, so break out of retry loop
                break
                
            except KeyboardInterrupt:
                print("\n\nReceived interrupt signal...")
                if stream is not None:
                    try:
                        if hasattr(stream, 'active') and stream.active:
                            stream.stop()
                        stream.close()
                    except:
                        pass
                break
            
            except Exception as e:
                error_msg = str(e)
                print(f"\nAttempt {attempt + 1}/{max_retries} - Error with audio stream: {error_msg}", file=sys.stderr)
                
                if stream is not None:
                    try:
                        if hasattr(stream, 'active') and stream.active:
                            stream.stop()
                        stream.close()
                    except:
                        pass
                    stream = None
                
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...", file=sys.stderr)
                    time.sleep(retry_delay)
                else:
                    print("\nFailed to start audio stream after all retries.", file=sys.stderr)
                    print("This may be due to:", file=sys.stderr)
                    print("  1. No microphone connected or enabled", file=sys.stderr)
                    print("  2. Microphone is in use by another application", file=sys.stderr)
                    print("  3. ALSA/PortAudio threading issues", file=sys.stderr)
                    print("  4. Try setting: export SDL_AUDIODRIVER=pulse", file=sys.stderr)
                    print("\nTry running: python scripts/streaming_transcribe.py --list-devices", file=sys.stderr)
        
        # Cleanup
        self.is_recording = False
        
        # Wait for processing thread to finish (with timeout)
        if processing_thread.is_alive():
            processing_thread.join(timeout=2.0)
        
        print("\nStream closed.")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Streaming audio transcription with sentence detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file (default: config/streaming_config.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Whisper model to use (overrides config file)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Audio sample rate in Hz (overrides config file)",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=None,
        help="Duration of audio chunks in seconds (overrides config file)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="Voice activity detection threshold (overrides config file)",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=None,
        help="Silence duration to consider end of sentence in seconds (overrides config file)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available (overrides config file)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code for transcription (e.g., 'en', 'es', 'fr'). Use 'auto' for auto-detection (overrides config file).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    
    args = parser.parse_args()
    
    # Load configuration from file
    config_file = args.config
    if config_file is None:
        # Try default location
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        default_config = project_dir / "config" / "streaming_config.yaml"
        if default_config.exists():
            config_file = str(default_config)
    
    # Set defaults from config file if it exists
    config_data = {}
    if config_file and Path(config_file).exists():
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            print(f"Loaded configuration from: {config_file}")
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            print("Using default values...")
    
    # Extract config values with defaults
    model_name = args.model if args.model else config_data.get('model', {}).get('name', 'openai/whisper-small')
    sample_rate = args.sample_rate if args.sample_rate else config_data.get('audio', {}).get('sample_rate', 16000)
    chunk_duration = args.chunk_duration if args.chunk_duration else config_data.get('audio', {}).get('chunk_duration', 3.0)
    vad_threshold = args.vad_threshold if args.vad_threshold else config_data.get('audio', {}).get('vad_threshold', 0.02)
    silence_duration = args.silence_duration if args.silence_duration else config_data.get('audio', {}).get('silence_duration', 1.5)
    use_gpu = args.gpu if args.gpu else config_data.get('model', {}).get('use_gpu', False)
    
    # Handle language setting
    language = args.language
    if language is None:
        config_lang = config_data.get('language', {}).get('code', None)
        if config_lang and config_lang.lower() != 'auto':
            language = config_lang
    elif language.lower() == 'auto':
        language = None
    
    # List devices if requested
    if args.list_devices:
        print("\nAvailable audio devices:")
        print("=" * 80)
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']}")
            print(f"   Input channels: {device['max_input_channels']}")
            print(f"   Output channels: {device['max_output_channels']}")
            print()
        return
    
    # Create transcriber
    transcriber = StreamingTranscriber(
        model_name=model_name,
        sample_rate=sample_rate,
        chunk_duration=chunk_duration,
        vad_threshold=vad_threshold,
        silence_duration=silence_duration,
        use_gpu=use_gpu,
        language=language,
    )
    
    # Start streaming
    transcriber.start_streaming()


if __name__ == "__main__":
    main()
