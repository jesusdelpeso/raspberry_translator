"""
Main translator class orchestrating the translation pipeline
"""

import numpy as np

from .audio_handler import AudioHandler
from .config import Config
from .models import ModelLoader


class RealTimeTranslator:
    """Main translator class handling the full pipeline"""

    def __init__(self, config: Config):
        self.config = config
        self.is_running = False

        # Initialize components
        self.audio_handler = AudioHandler(
            sample_rate=config.sample_rate,
            channels=config.channels,
        )
        self.model_loader = ModelLoader(config)

        # Load models
        self.stt_pipe = self.model_loader.load_stt_model()
        self.translator = self.model_loader.load_translation_model()
        self.tts_pipe = self.model_loader.load_tts_model()

        print("All models loaded successfully!")

    def process_audio_chunk(self, audio_data):
        """
        Process a chunk of audio through the full pipeline
        
        Args:
            audio_data: Audio data as numpy array
        """
        try:
            # Convert to the right format for the model
            audio_float = audio_data.flatten().astype(np.float32)

            # Speech-to-Text
            print("Transcribing...")
            result = self.stt_pipe(audio_float, sampling_rate=self.config.sample_rate)
            text = result["text"].strip()

            if not text:
                print("No speech detected")
                return

            print(f"Detected: {text}")

            # Translation
            print("Translating...")
            translation_result = self.translator(text)
            translated_text = translation_result[0]["translation_text"]
            print(f"Translated: {translated_text}")

            # Text-to-Speech
            print("Generating speech...")
            speech_output = self.tts_pipe(translated_text)

            # Play the generated audio
            self.play_audio(speech_output)

        except Exception as e:
            print(f"Error processing audio: {e}")

    def play_audio(self, speech_output):
        """Play the generated audio"""
        try:
            audio_array = speech_output["audio"]
            sampling_rate = speech_output["sampling_rate"]

            self.audio_handler.play_audio(audio_array, sampling_rate)

        except Exception as e:
            print(f"Error playing audio: {e}")

    def start_listening(self):
        """Start listening to microphone"""
        print(f"\nListening... (Press Ctrl+C to stop)")
        print(f"Source language: {self.config.source_lang}")
        print(f"Target language: {self.config.target_lang}")
        print(f"Recording duration: {self.config.recording_duration}s chunks\n")

        self.is_running = True

        try:
            while self.is_running:
                # Record audio for specified duration
                audio_data = self.audio_handler.record_audio(
                    self.config.recording_duration
                )

                # Process the recorded audio
                self.process_audio_chunk(audio_data)
                print("-" * 50)

        except KeyboardInterrupt:
            print("\nStopping translator...")
            self.is_running = False

    def stop(self):
        """Stop the translator"""
        self.is_running = False
