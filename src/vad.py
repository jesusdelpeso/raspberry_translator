"""
Voice Activity Detection using Silero VAD.

Only audio chunks that contain speech are passed to the translation
pipeline, avoiding wasted compute on silence.
"""

import logging

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from .config import Config

logger = logging.getLogger(__name__)

# Silero VAD supports only these two sample rates.
_SUPPORTED_SAMPLE_RATES = {8000, 16000}


class VADDetector:
    """Detects whether an audio chunk contains speech using Silero VAD."""

    def __init__(self, config: Config):
        if config.sample_rate not in _SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"VAD requires a sample rate of 8000 or 16000 Hz, "
                f"but config.sample_rate={config.sample_rate}. "
                f"Set vad_enabled: false or change sample_rate."
            )

        self.threshold = config.vad_threshold
        self.sample_rate = config.sample_rate

        logger.info("Loading VAD model...")
        self.model = load_silero_vad()
        logger.info("VAD model loaded.")

    def has_speech(self, audio_data: np.ndarray) -> bool:
        """
        Return True if speech is detected in the audio chunk.

        Args:
            audio_data: numpy array produced by AudioHandler.record_audio(),
                        shape (N,) or (N, channels), dtype float32.

        Returns:
            True if at least one speech segment is detected.
        """
        audio_tensor = torch.from_numpy(audio_data.flatten().astype(np.float32))
        timestamps = get_speech_timestamps(
            audio_tensor,
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
        )
        return len(timestamps) > 0
