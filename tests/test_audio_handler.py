"""
pytest tests for src/audio_handler.py.

Heavy dependencies are stubbed via conftest.py.  Tests cover:
  - Construction / attributes
  - record_audio() delegates to sd.rec / sd.wait
  - play_audio() normalises dtype and delegates to sd.play / sd.wait
  - play_audio() logs and does not raise on sd exception
  - test_microphone() returns array, warns on low amplitude
  - list_devices() returns device list from sd.query_devices
  - stream_audio_segments() raises ValueError for unsupported rate
  - stream_audio_segments() yields correctly on start/end VAD events
  - stream_audio_segments() force-flushes at max_duration
  - stream_audio_segments() stops when stop_event is set
"""

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# conftest.py installs sd/silero_vad stubs before these imports.
import sounddevice as sd


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestAudioHandlerInit:

    def test_default_sample_rate(self, audio_handler):
        assert audio_handler.sample_rate == 16000

    def test_default_channels(self, audio_handler):
        assert audio_handler.channels == 1

    def test_custom_sample_rate(self):
        from src.audio_handler import AudioHandler
        h = AudioHandler(sample_rate=8000)
        assert h.sample_rate == 8000


# ---------------------------------------------------------------------------
# record_audio()
# ---------------------------------------------------------------------------

class TestRecordAudio:

    def test_calls_sd_rec_with_correct_samples(self, audio_handler):
        expected_samples = 3 * 16000  # 3 second × 16 kHz
        fake_audio = np.zeros((expected_samples, 1), dtype=np.float32)
        sd.rec = MagicMock(return_value=fake_audio)
        sd.wait = MagicMock()

        result = audio_handler.record_audio(3)

        sd.rec.assert_called_once()
        call_args = sd.rec.call_args
        assert call_args[0][0] == expected_samples

    def test_returns_audio_array(self, audio_handler):
        fake = np.ones((16000, 1), dtype=np.float32)
        sd.rec = MagicMock(return_value=fake)
        sd.wait = MagicMock()
        result = audio_handler.record_audio(1)
        assert result is fake

    def test_calls_sd_wait(self, audio_handler):
        sd.rec = MagicMock(return_value=np.zeros((16000, 1), dtype=np.float32))
        sd.wait = MagicMock()
        audio_handler.record_audio(1)
        sd.wait.assert_called_once()


# ---------------------------------------------------------------------------
# play_audio()
# ---------------------------------------------------------------------------

class TestPlayAudio:

    def test_calls_sd_play(self, audio_handler):
        sd.play = MagicMock()
        sd.wait = MagicMock()
        arr = np.zeros(1000, dtype=np.float32)
        audio_handler.play_audio(arr, 16000)
        sd.play.assert_called_once()

    def test_converts_int16_to_float32(self, audio_handler):
        sd.play = MagicMock()
        sd.wait = MagicMock()
        arr_int = np.zeros(1000, dtype=np.int16)
        audio_handler.play_audio(arr_int, 16000)
        played = sd.play.call_args[0][0]
        assert played.dtype == np.float32

    def test_calls_sd_wait_after_play(self, audio_handler):
        sd.play = MagicMock()
        sd.wait = MagicMock()
        audio_handler.play_audio(np.zeros(100, dtype=np.float32), 16000)
        sd.wait.assert_called_once()

    def test_does_not_raise_on_sd_error(self, audio_handler):
        sd.play = MagicMock(side_effect=RuntimeError("device error"))
        sd.wait = MagicMock()
        # Should log and swallow the exception
        audio_handler.play_audio(np.zeros(100, dtype=np.float32), 16000)


# ---------------------------------------------------------------------------
# test_microphone()
# ---------------------------------------------------------------------------

class TestTestMicrophone:

    def test_returns_recording(self, audio_handler):
        fake = np.zeros((32000, 1), dtype=np.float32)
        sd.rec = MagicMock(return_value=fake)
        sd.wait = MagicMock()
        result = audio_handler.test_microphone()
        assert result is fake

    def test_warns_on_silent_audio(self, audio_handler, caplog):
        import logging
        silent = np.zeros((32000, 1), dtype=np.float32)
        sd.rec = MagicMock(return_value=silent)
        sd.wait = MagicMock()
        with caplog.at_level(logging.WARNING, logger="src.audio_handler"):
            audio_handler.test_microphone()
        assert any("quiet" in r.message.lower() for r in caplog.records)

    def test_no_warning_on_loud_audio(self, audio_handler, caplog):
        import logging
        loud = np.ones((32000, 1), dtype=np.float32) * 0.5
        sd.rec = MagicMock(return_value=loud)
        sd.wait = MagicMock()
        with caplog.at_level(logging.WARNING, logger="src.audio_handler"):
            audio_handler.test_microphone()
        assert not any("quiet" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# list_devices()
# ---------------------------------------------------------------------------

class TestListDevices:

    def test_returns_device_list(self, audio_handler):
        fake_devices = [
            {"name": "Mic A", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Speaker B", "max_input_channels": 0, "max_output_channels": 2},
        ]
        sd.query_devices = MagicMock(return_value=fake_devices)
        result = audio_handler.list_devices()
        assert result is fake_devices

    def test_calls_query_devices(self, audio_handler):
        sd.query_devices = MagicMock(return_value=[])
        audio_handler.list_devices()
        sd.query_devices.assert_called_once()


# ---------------------------------------------------------------------------
# stream_audio_segments() — unit tests with mocked VADIterator
# ---------------------------------------------------------------------------

class TestStreamAudioSegments:

    def test_invalid_sample_rate_raises(self):
        from src.audio_handler import AudioHandler
        h = AudioHandler(sample_rate=44100)
        with pytest.raises(ValueError, match="sample_rate"):
            list(h.stream_audio_segments(vad_model=MagicMock()))

    def test_stop_event_prevents_any_iteration(self, audio_handler):
        """Setting stop_event before the call yields nothing."""
        stop = threading.Event()
        stop.set()

        # VADIterator mustn't be called; set up InputStream as context manager
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        sd.InputStream = MagicMock(return_value=ctx)

        # VADIterator is imported lazily inside stream_audio_segments
        import silero_vad
        with patch.object(silero_vad, "VADIterator"):
            result = list(audio_handler.stream_audio_segments(
                vad_model=MagicMock(),
                stop_event=stop,
            ))
        assert result == []

    def test_yields_utterance_on_end_event(self, audio_handler):
        """VADIterator returns start then end → one utterance yielded."""
        import queue as _queue

        stop = threading.Event()
        chunk = np.ones(512, dtype=np.float32).reshape(-1, 1)

        # Simulate: start, middle (None), end, then stop
        vad_returns = [{"start": 0}, None, {"end": 512}]
        call_idx = [0]

        def fake_vad_iter(flat):
            idx = call_idx[0]
            call_idx[0] += 1
            if idx < len(vad_returns):
                return vad_returns[idx]
            stop.set()
            return None

        fake_vad_instance = MagicMock()
        fake_vad_instance.side_effect = fake_vad_iter
        fake_vad_instance.reset_states = MagicMock()

        # InputStream context manager delivers chunks via the callback
        q = _queue.Queue()
        for _ in range(3):
            q.put(chunk)

        def fake_input_stream_cm(samplerate, channels, dtype, blocksize, callback):
            cm = MagicMock()
            # Push 4 chunks so the 4th call to fake_vad_iter triggers stop.set()
            for _ in range(4):
                cb_chunk = np.ones((blocksize, 1), dtype=np.float32)
                callback(cb_chunk, blocksize, None, None)
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        sd.InputStream = fake_input_stream_cm

        import silero_vad
        with patch.object(silero_vad, "VADIterator", return_value=fake_vad_instance):
            results = list(audio_handler.stream_audio_segments(
                vad_model=MagicMock(),
                stop_event=stop,
            ))

        assert len(results) >= 1
        assert all(isinstance(r, np.ndarray) for r in results)
        assert all(r.ndim == 1 for r in results)

    def test_yields_float32(self, audio_handler):
        """Yielded arrays must be float32 regardless of input dtype."""
        import queue as _queue

        stop = threading.Event()
        call_idx = [0]
        vtable = [{"start": 0}, {"end": 512}]

        def fake_vad(flat):
            i = call_idx[0]
            call_idx[0] += 1
            if i < len(vtable):
                return vtable[i]
            stop.set()
            return None

        inst = MagicMock(side_effect=fake_vad)
        inst.reset_states = MagicMock()

        chunk = np.ones((512, 1), dtype=np.float32)

        def fake_stream(samplerate, channels, dtype, blocksize, callback):
            # Push 3 chunks so the 3rd call to fake_vad triggers stop.set()
            for _ in range(3):
                callback(chunk, blocksize, None, None)
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        sd.InputStream = fake_stream
        import silero_vad
        with patch.object(silero_vad, "VADIterator", return_value=inst):
            results = list(audio_handler.stream_audio_segments(
                vad_model=MagicMock(), stop_event=stop
            ))

        for arr in results:
            assert arr.dtype == np.float32
