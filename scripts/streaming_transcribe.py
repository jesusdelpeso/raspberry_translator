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
        
        language_msg = f" (language: {self.language})" if self.language else " (auto-detect language)"
        print(f"Model loaded successfully!{language_msg}\n")

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
            # Transcribe
            result = self.pipe(audio_array)
            text = result["text"].strip()
            return text if text else None
        except Exception as e:
            print(f"Transcription error: {e}", file=sys.stderr)
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
        
        # Start audio stream
        stream = None
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.chunk_samples,
                callback=self.audio_callback,
            ) as stream:
                # Wait for processing thread or KeyboardInterrupt
                while self.is_recording and processing_thread.is_alive():
                    time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n\nReceived interrupt signal...")
        
        except Exception as e:
            print(f"\nError with audio stream: {e}", file=sys.stderr)
        
        finally:
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
        "--model",
        type=str,
        default="openai/whisper-small",
        help="Whisper model to use (default: openai/whisper-small)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=3.0,
        help="Duration of audio chunks in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.02,
        help="Voice activity detection threshold (default: 0.02)",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=1.5,
        help="Silence duration to consider end of sentence in seconds (default: 1.5)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code for transcription (e.g., 'en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'pl', 'ru', 'zh', 'ja', 'ko'). If not specified, language will be auto-detected.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    
    args = parser.parse_args()
    
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
        model_name=args.model,
        sample_rate=args.sample_rate,
        chunk_duration=args.chunk_duration,
        vad_threshold=args.vad_threshold,
        silence_duration=args.silence_duration,
        use_gpu=args.gpu,
        language=args.language,
    )
    
    # Start streaming
    transcriber.start_streaming()


if __name__ == "__main__":
    main()
