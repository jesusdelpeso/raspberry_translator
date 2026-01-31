"""
Audio input/output handling
"""

import numpy as np
import sounddevice as sd


class AudioHandler:
    """Handles audio recording and playback"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels

    def record_audio(self, duration: int):
        """
        Record audio from microphone
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            numpy array of audio data
        """
        print("Recording...")
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
        )
        sd.wait()
        return audio_data

    def play_audio(self, audio_array, sampling_rate: int):
        """
        Play audio through speakers
        
        Args:
            audio_array: Audio data as numpy array
            sampling_rate: Sample rate of the audio
        """
        try:
            # Normalize audio
            audio_array = np.array(audio_array)
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)

            # Play audio
            sd.play(audio_array, sampling_rate)
            sd.wait()

        except Exception as e:
            print(f"Error playing audio: {e}")

    def test_microphone(self):
        """Test microphone by recording a short sample"""
        print("Testing microphone (2 seconds)...")
        recording = self.record_audio(2)
        
        max_amplitude = np.max(np.abs(recording))
        print(f"Recording successful! Max amplitude: {max_amplitude:.4f}")
        
        if max_amplitude < 0.001:
            print("⚠ Warning: Very quiet audio. Check microphone levels.")
        
        return recording

    @staticmethod
    def list_devices():
        """List all available audio devices"""
        devices = sd.query_devices()
        print("\nAvailable audio devices:")
        print("-" * 80)
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']}")
            print(f"   Input channels: {device['max_input_channels']}")
            print(f"   Output channels: {device['max_output_channels']}")
            print()
        
        return devices
