"""
Audio input/output handling
"""

import logging
import queue
import threading
from collections import deque
from typing import Generator, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# Silero VAD requires exactly these chunk sizes.
_VAD_CHUNK_SAMPLES = {16000: 512, 8000: 256}


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
        logger.debug("Recording %s seconds of audio...", duration)
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
            logger.error("Error playing audio: %s", e)

    def stream_audio_segments(
        self,
        vad_model,
        threshold: float = 0.5,
        min_silence_ms: int = 600,
        max_duration_s: float = 30.0,
        speech_pad_ms: int = 30,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[np.ndarray, None, None]:
        """
        Generator that continuously captures microphone audio and yields one
        numpy array per detected utterance, segmenting at natural speech
        boundaries using Silero VADIterator.

        The generator runs until ``stop_event`` is set or a KeyboardInterrupt
        is raised.  It should be consumed inside a try/except KeyboardInterrupt
        block.

        Args:
            vad_model:      Silero VAD model returned by ``load_silero_vad()``.
            threshold:      Speech-probability threshold (0–1).
            min_silence_ms: Milliseconds of silence that signals utterance end.
            max_duration_s: Hard cap on utterance length; forces a flush.
            speech_pad_ms:  Padding (ms) added by VADIterator around segments.
            stop_event:     Optional ``threading.Event``; set it to stop the
                            generator cleanly from another thread.

        Yields:
            numpy.ndarray of shape ``(N,)``, dtype ``float32``,
            at ``self.sample_rate`` Hz.

        Raises:
            ValueError: If the configured sample rate is not 8000 or 16000 Hz.
        """
        from silero_vad import VADIterator

        if self.sample_rate not in _VAD_CHUNK_SAMPLES:
            raise ValueError(
                f"Streaming STT requires sample_rate 8000 or 16000 Hz, "
                f"got {self.sample_rate}."
            )

        chunk_samples = _VAD_CHUNK_SAMPLES[self.sample_rate]
        max_samples = int(max_duration_s * self.sample_rate)

        vad_iterator = VADIterator(
            vad_model,
            threshold=threshold,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )

        audio_queue: queue.Queue = queue.Queue()

        def _callback(indata, frames, time_info, status):
            audio_queue.put(indata.copy())

        # Keep ~96 ms of audio before detected speech start so the first
        # syllable is not clipped.
        pre_roll: deque = deque(maxlen=3)
        speech_buffer: list = []
        in_speech: bool = False

        logger.debug("Streaming: listening for speech boundaries...")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            callback=_callback,
        ):
            while not (stop_event and stop_event.is_set()):
                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                flat = chunk.flatten()
                vad_result = vad_iterator(flat)

                if vad_result is None:
                    if in_speech:
                        speech_buffer.append(flat)
                        # Force-flush if the utterance exceeds max duration.
                        if sum(len(b) for b in speech_buffer) >= max_samples:
                            utterance = np.concatenate(speech_buffer)
                            in_speech = False
                            speech_buffer = []
                            pre_roll.clear()
                            vad_iterator.reset_states()
                            logger.debug(
                                "Max duration reached, flushing (%.1fs).",
                                len(utterance) / self.sample_rate,
                            )
                            yield utterance
                    else:
                        pre_roll.append(flat)

                elif "start" in vad_result:
                    # Include pre-roll so the very beginning of speech is kept.
                    pre_roll.append(flat)
                    in_speech = True
                    speech_buffer = list(pre_roll)
                    logger.debug("Speech start detected.")

                elif "end" in vad_result and in_speech:
                    speech_buffer.append(flat)
                    utterance = np.concatenate(speech_buffer)
                    in_speech = False
                    speech_buffer = []
                    pre_roll.clear()
                    vad_iterator.reset_states()
                    logger.debug(
                        "Speech end detected (%.1fs).",
                        len(utterance) / self.sample_rate,
                    )
                    yield utterance

    def test_microphone(self):
        """Test microphone by recording a short sample"""
        logger.info("Testing microphone (2 seconds)...")
        recording = self.record_audio(2)
        
        max_amplitude = np.max(np.abs(recording))
        logger.info("Recording successful! Max amplitude: %.4f", max_amplitude)
        
        if max_amplitude < 0.001:
            logger.warning("Very quiet audio. Check microphone levels.")
        
        return recording

    @staticmethod
    def list_devices():
        """List all available audio devices"""
        devices = sd.query_devices()
        lines = ["\nAvailable audio devices:", "-" * 80]
        for i, device in enumerate(devices):
            lines.append(f"{i}: {device['name']}")
            lines.append(f"   Input channels: {device['max_input_channels']}")
            lines.append(f"   Output channels: {device['max_output_channels']}")
            lines.append("")
        logger.info("\n".join(lines))
        return devices
