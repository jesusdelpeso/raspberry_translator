"""
Tests for TODO 10 — Graceful error recovery.

Covers:
  - retry(): success on first try
  - retry(): succeeds after N-1 failures
  - retry(): exhausts all attempts → RetryError
  - retry(): non-matching exception re-raised immediately (no sleep)
  - retry(): max_attempts=1 means no retry
  - retry(): max_attempts<1 raises ValueError
  - retry(): last_exception attribute set correctly on RetryError
  - retry(): wait time grows with backoff (sleep calls)
  - retry(): wait capped at max_delay_s
  - Config recovery field defaults
  - Config YAML round-trip for recovery section
  - RealTimeTranslator.__init__ retries model loads then raises RuntimeError
  - main() exits with code 1 on model load failure
  - _start_streaming() reconnects on audio error up to audio_device_retries
  - _start_streaming() stops after exhausting reconnect attempts
  - _start_fixed_chunks() retries on audio error, resets on success
  - _start_fixed_chunks() stops on too many consecutive errors
"""

import sys
import time
import types
import unittest
from unittest.mock import MagicMock, call, patch

# ---------------------------------------------------------------------------
# Stubs (before any src import)
# ---------------------------------------------------------------------------

_sd = types.ModuleType("sounddevice")
_sd.rec = lambda *a, **kw: None
_sd.wait = lambda: None
_sd.play = lambda *a, **kw: None
_sd.InputStream = MagicMock
sys.modules.setdefault("sounddevice", _sd)

_sv = types.ModuleType("silero_vad")
_sv.load_silero_vad = lambda: MagicMock()
_sv.get_speech_timestamps = lambda *a, **kw: []
_sv.VADIterator = MagicMock
sys.modules.setdefault("silero_vad", _sv)

for _mod in ("torch", "transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import numpy as np  # noqa: E402  (available in venv)


# ---------------------------------------------------------------------------
# retry() unit tests
# ---------------------------------------------------------------------------

class TestRetrySuccess(unittest.TestCase):

    def test_returns_on_first_success(self):
        from src.recovery import retry
        fn = MagicMock(return_value=42)
        result = retry(fn, max_attempts=3, delay_s=0)
        self.assertEqual(result, 42)
        fn.assert_called_once()

    def test_success_after_one_failure(self):
        from src.recovery import retry
        fn = MagicMock(side_effect=[RuntimeError("oops"), "ok"])
        result = retry(fn, max_attempts=3, delay_s=0)
        self.assertEqual(result, "ok")
        self.assertEqual(fn.call_count, 2)

    def test_success_on_last_attempt(self):
        from src.recovery import retry
        fn = MagicMock(side_effect=[ValueError("a"), ValueError("b"), "done"])
        result = retry(fn, max_attempts=3, delay_s=0)
        self.assertEqual(result, "done")
        self.assertEqual(fn.call_count, 3)

    def test_passes_args_and_kwargs(self):
        from src.recovery import retry
        fn = MagicMock(return_value=7)
        retry(fn, 1, 2, max_attempts=1, delay_s=0, key="val")
        fn.assert_called_once_with(1, 2, key="val")


class TestRetryExhausted(unittest.TestCase):

    def test_raises_retry_error_after_all_attempts(self):
        from src.recovery import RetryError, retry
        fn = MagicMock(side_effect=IOError("disk error"))
        with self.assertRaises(RetryError):
            retry(fn, max_attempts=3, delay_s=0)
        self.assertEqual(fn.call_count, 3)

    def test_last_exception_attribute(self):
        from src.recovery import RetryError, retry
        exc = OSError("device gone")
        fn = MagicMock(side_effect=exc)
        with self.assertRaises(RetryError) as ctx:
            retry(fn, max_attempts=2, delay_s=0)
        self.assertIs(ctx.exception.last_exception, exc)

    def test_max_attempts_1_no_retry(self):
        from src.recovery import RetryError, retry
        fn = MagicMock(side_effect=ValueError("x"))
        with self.assertRaises(RetryError):
            retry(fn, max_attempts=1, delay_s=0)
        fn.assert_called_once()

    def test_invalid_max_attempts_raises_value_error(self):
        from src.recovery import retry
        with self.assertRaises(ValueError):
            retry(lambda: None, max_attempts=0, delay_s=0)

    def test_invalid_max_attempts_negative(self):
        from src.recovery import retry
        with self.assertRaises(ValueError):
            retry(lambda: None, max_attempts=-1, delay_s=0)


class TestRetryExceptionFilter(unittest.TestCase):

    def test_non_matching_exception_reraised_immediately(self):
        """An exception type not in *exceptions* must propagate without retry."""
        from src.recovery import retry
        fn = MagicMock(side_effect=KeyError("unexpected"))
        with self.assertRaises(KeyError):
            retry(fn, max_attempts=5, delay_s=0, exceptions=(OSError,))
        # Called only once — no retry for non-matching type.
        fn.assert_called_once()

    def test_matching_exception_retried(self):
        from src.recovery import retry
        fn = MagicMock(side_effect=[OSError("retry me"), "success"])
        result = retry(fn, max_attempts=3, delay_s=0, exceptions=(OSError,))
        self.assertEqual(result, "success")

    def test_subclass_exception_retried(self):
        from src.recovery import retry
        fn = MagicMock(side_effect=[FileNotFoundError("sub"), "ok"])
        result = retry(fn, max_attempts=3, delay_s=0, exceptions=(OSError,))
        self.assertEqual(result, "ok")


class TestRetryBackoff(unittest.TestCase):

    @patch("src.recovery.time.sleep")
    def test_sleep_called_between_attempts(self, mock_sleep):
        from src.recovery import retry
        fn = MagicMock(side_effect=[RuntimeError(), RuntimeError(), "ok"])
        retry(fn, max_attempts=3, delay_s=1.0, backoff=2.0)
        # Should sleep twice: 1.0s and 2.0s
        mock_sleep.assert_has_calls([call(1.0), call(2.0)])

    @patch("src.recovery.time.sleep")
    def test_no_sleep_on_first_attempt_success(self, mock_sleep):
        from src.recovery import retry
        fn = MagicMock(return_value="ok")
        retry(fn, max_attempts=3, delay_s=5.0)
        mock_sleep.assert_not_called()

    @patch("src.recovery.time.sleep")
    def test_delay_capped_at_max_delay_s(self, mock_sleep):
        from src.recovery import retry
        # Three failures: waits should be 1.0, min(2.0, 1.5)=1.5 — but cap=1.5
        fn = MagicMock(
            side_effect=[RuntimeError(), RuntimeError(), RuntimeError(), "ok"]
        )
        retry(fn, max_attempts=4, delay_s=1.0, backoff=2.0, max_delay_s=1.5)
        calls = [c[0][0] for c in mock_sleep.call_args_list]
        self.assertLessEqual(max(calls), 1.5)


# ---------------------------------------------------------------------------
# Config recovery fields
# ---------------------------------------------------------------------------

class TestConfigRecoveryFields(unittest.TestCase):

    def test_defaults(self):
        from src.config import Config
        c = Config()
        self.assertEqual(c.model_load_retries, 3)
        self.assertAlmostEqual(c.model_load_retry_delay_s, 5.0)
        self.assertEqual(c.audio_device_retries, 5)
        self.assertAlmostEqual(c.audio_device_retry_delay_s, 2.0)

    def test_yaml_round_trip(self):
        import os
        import tempfile

        import yaml

        from src.config import Config

        c = Config(
            model_load_retries=1,
            model_load_retry_delay_s=0.5,
            audio_device_retries=2,
            audio_device_retry_delay_s=1.0,
        )
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            c.save_yaml(path)
            c2 = Config.from_yaml(path)
            self.assertEqual(c2.model_load_retries, 1)
            self.assertAlmostEqual(c2.model_load_retry_delay_s, 0.5)
            self.assertEqual(c2.audio_device_retries, 2)
            self.assertAlmostEqual(c2.audio_device_retry_delay_s, 1.0)
        finally:
            os.unlink(path)

    def test_yaml_missing_recovery_section_uses_defaults(self):
        import os
        import tempfile

        import yaml

        from src.config import Config

        minimal = {"models": {}, "languages": {}, "audio": {}, "performance": {}}
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False, mode="w"
        ) as f:
            yaml.dump(minimal, f)
            path = f.name
        try:
            c = Config.from_yaml(path)
            self.assertEqual(c.model_load_retries, 3)
            self.assertEqual(c.audio_device_retries, 5)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# RealTimeTranslator model load retry via __init__
# ---------------------------------------------------------------------------

def _make_translator_mocks():
    """Return a Config and patched ModelLoader/VADDetector for translator tests."""
    from src.config import Config

    config = Config(
        model_load_retries=2,
        model_load_retry_delay_s=0,  # no delay in tests
        audio_device_retries=3,
        audio_device_retry_delay_s=0,
        vad_enabled=False,
        streaming_enabled=False,
    )
    return config


class TestTranslatorModelLoadRetry(unittest.TestCase):

    @patch("src.translator.VADDetector", return_value=MagicMock())
    @patch("src.translator.ConversationHistory", return_value=MagicMock())
    @patch("src.translator.AudioHandler", return_value=MagicMock())
    @patch("src.translator.ModelLoader")
    def test_model_retried_then_succeeds(
        self, MockLoader, _ah, _ch, _vad
    ):
        """Model load fails once then succeeds — no RuntimeError raised."""
        from src.config import Config
        from src.translator import RealTimeTranslator

        loader = MagicMock()
        loader.load_stt_model.side_effect = [OSError("network"), MagicMock()]
        loader.load_translation_model.return_value = MagicMock()
        loader.load_tts_model.return_value = MagicMock()
        MockLoader.return_value = loader

        config = Config(
            model_load_retries=3,
            model_load_retry_delay_s=0,
            vad_enabled=False,
            streaming_enabled=False,
        )
        t = RealTimeTranslator(config)
        self.assertEqual(loader.load_stt_model.call_count, 2)

    @patch("src.translator.VADDetector", return_value=MagicMock())
    @patch("src.translator.ConversationHistory", return_value=MagicMock())
    @patch("src.translator.AudioHandler", return_value=MagicMock())
    @patch("src.translator.ModelLoader")
    def test_model_load_exhausted_raises_runtime_error(
        self, MockLoader, _ah, _ch, _vad
    ):
        """All retry attempts fail → RuntimeError with clear message."""
        from src.config import Config
        from src.translator import RealTimeTranslator

        loader = MagicMock()
        loader.load_stt_model.side_effect = OSError("connection refused")
        MockLoader.return_value = loader

        config = Config(
            model_load_retries=2,
            model_load_retry_delay_s=0,
            vad_enabled=False,
            streaming_enabled=False,
        )
        with self.assertRaises(RuntimeError) as ctx:
            RealTimeTranslator(config)
        self.assertIn("Failed to load required model", str(ctx.exception))
        # Called model_load_retries times
        self.assertEqual(loader.load_stt_model.call_count, 2)


# ---------------------------------------------------------------------------
# main() exits with code 1 on RuntimeError from translator
# ---------------------------------------------------------------------------

class TestMainHandlesStartupFailure(unittest.TestCase):

    def test_returns_1_on_model_load_failure(self):
        from src.config import Config

        with (
            patch("src.main.setup_logging"),
            patch("src.main.validate_config") as mock_val,
            patch("src.main.print_validation_report"),
            patch("src.main.is_interactive", return_value=False),
            patch(
                "src.main.RealTimeTranslator",
                side_effect=RuntimeError("model failed"),
            ),
            patch("sys.argv", ["translator"]),
        ):
            mock_result = MagicMock()
            mock_result.ok = True
            mock_result.warnings = []
            mock_val.return_value = mock_result

            from src.main import main

            exit_code = main()

        self.assertEqual(exit_code, 1)


# ---------------------------------------------------------------------------
# _start_streaming() audio device reconnect
# ---------------------------------------------------------------------------

class TestStreamingReconnect(unittest.TestCase):

    def _make_translator(self, audio_device_retries=3, audio_device_retry_delay_s=0):
        from src.config import Config
        from src.translator import RealTimeTranslator

        with (
            patch("src.translator.ModelLoader") as MockLoader,
            patch("src.translator.AudioHandler"),
            patch("src.translator.ConversationHistory"),
            patch("src.translator.VADDetector"),
        ):
            loader = MagicMock()
            loader.load_stt_model.return_value = MagicMock()
            loader.load_translation_model.return_value = MagicMock()
            loader.load_tts_model.return_value = MagicMock()
            MockLoader.return_value = loader

            config = Config(
                model_load_retries=1,
                model_load_retry_delay_s=0,
                audio_device_retries=audio_device_retries,
                audio_device_retry_delay_s=audio_device_retry_delay_s,
                vad_enabled=False,
                streaming_enabled=True,
            )
            t = RealTimeTranslator(config)

        t.is_running = True
        t.vad = MagicMock()
        t.history = MagicMock()
        t.process_audio_chunk = MagicMock()
        return t

    @patch("src.translator.time.sleep")
    def test_reconnects_on_audio_error(self, mock_sleep):
        """Stream raises once, recovers on second attempt, then stops."""
        t = self._make_translator(audio_device_retries=2)

        call_count = [0]

        def stream_gen(**kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError("device lost")
            # Second call: yield one chunk then return
            yield np.zeros(512, dtype=np.float32)
            t.is_running = False

        t.audio_handler.stream_audio_segments = stream_gen
        t._start_streaming()

        self.assertEqual(call_count[0], 2)
        t.process_audio_chunk.assert_called_once()

    @patch("src.translator.time.sleep")
    def test_stops_after_max_reconnects(self, mock_sleep):
        """Stream always errors → stops after audio_device_retries exhausted."""
        t = self._make_translator(audio_device_retries=3)

        call_count = [0]

        def always_fail(**kw):
            call_count[0] += 1
            raise OSError("gone")

        t.audio_handler.stream_audio_segments = always_fail
        t._start_streaming()

        # max_reconnects=3 → 4 total attempts (initial + 3 reconnects)
        self.assertEqual(call_count[0], 4)
        t.process_audio_chunk.assert_not_called()

    @patch("src.translator.time.sleep")
    def test_sets_is_running_false_after_exhausted(self, mock_sleep):
        t = self._make_translator(audio_device_retries=1)
        t.audio_handler.stream_audio_segments = MagicMock(
            side_effect=OSError("gone")
        )
        t._start_streaming()
        self.assertFalse(t.is_running)


# ---------------------------------------------------------------------------
# _start_fixed_chunks() audio device reconnect
# ---------------------------------------------------------------------------

class TestFixedChunksReconnect(unittest.TestCase):

    def _make_translator(self, audio_device_retries=3, audio_device_retry_delay_s=0):
        from src.config import Config
        from src.translator import RealTimeTranslator

        with (
            patch("src.translator.ModelLoader") as MockLoader,
            patch("src.translator.AudioHandler"),
            patch("src.translator.ConversationHistory"),
            patch("src.translator.VADDetector"),
        ):
            loader = MagicMock()
            loader.load_stt_model.return_value = MagicMock()
            loader.load_translation_model.return_value = MagicMock()
            loader.load_tts_model.return_value = MagicMock()
            MockLoader.return_value = loader

            config = Config(
                model_load_retries=1,
                model_load_retry_delay_s=0,
                audio_device_retries=audio_device_retries,
                audio_device_retry_delay_s=audio_device_retry_delay_s,
                vad_enabled=False,
                streaming_enabled=False,
            )
            t = RealTimeTranslator(config)

        t.is_running = True
        t.vad = None
        t.history = MagicMock()
        t.process_audio_chunk = MagicMock()
        return t

    @patch("src.translator.time.sleep")
    def test_retries_on_audio_error(self, mock_sleep):
        """record_audio fails once, succeeds on second call."""
        t = self._make_translator(audio_device_retries=3)

        call_count = [0]

        def record(duration):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError("device unavailable")
            t.is_running = False  # stop after first success
            return np.zeros((duration * 16000, 1), dtype=np.float32)

        t.audio_handler.record_audio = record
        t._start_fixed_chunks()

        self.assertEqual(call_count[0], 2)
        t.process_audio_chunk.assert_called_once()

    @patch("src.translator.time.sleep")
    def test_stops_after_too_many_consecutive_errors(self, mock_sleep):
        """record_audio always errors → stops after audio_device_retries."""
        t = self._make_translator(audio_device_retries=3)
        t.audio_handler.record_audio = MagicMock(side_effect=OSError("gone"))
        t._start_fixed_chunks()

        # max_reconnects=3 → 4 total calls (initial + 3 retries)
        self.assertEqual(t.audio_handler.record_audio.call_count, 4)
        t.process_audio_chunk.assert_not_called()

    @patch("src.translator.time.sleep")
    def test_error_counter_resets_on_success(self, mock_sleep):
        """Errors then success resets the consecutive error counter."""
        t = self._make_translator(audio_device_retries=2)

        call_count = [0]

        def record(duration):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise OSError("flaky")
            if call_count[0] == 3:
                return np.zeros(512, dtype=np.float32)
            # two more errors after the reset — still under max
            if call_count[0] <= 5:
                raise OSError("flaky again")
            t.is_running = False
            return np.zeros(512, dtype=np.float32)

        t.audio_handler.record_audio = record
        t._start_fixed_chunks()
        # Should not have stopped due to consecutive error limit
        self.assertFalse(t.is_running)  # stopped by t.is_running = False


if __name__ == "__main__":
    unittest.main(verbosity=2)
