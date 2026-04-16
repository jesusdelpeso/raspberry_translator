"""
pytest tests for the translation pipeline (RealTimeTranslator).

All heavy models are mocked.  Tests cover:
  - process_audio_chunk(): full happy path (STT → translate → TTS → play)
  - process_audio_chunk(): empty transcription does not call translator
  - process_audio_chunk(): exception in STT is caught and logged
  - process_audio_chunk(): exception in TTS is caught and logged
  - process_audio_chunk(): history.add() called with correct fields
  - play_audio(): delegates to audio_handler.play_audio
  - play_audio(): exception logged and not raised
  - start_listening() / stop(): sets is_running correctly
  - _same_language() helper
  - _set_translation_target() sets forced_bos_token_id
  - Bidirectional manual alternation toggles direction
  - auto_detect_source_lang=False uses source_lang without calling detect_language
"""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest


# conftest.py provides: default_config, mock_stt_pipe, mock_translation_pipe,
#                       mock_tts_pipe, translator


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_audio(samples: int = 512, sr: int = 16000) -> np.ndarray:
    return np.zeros((samples, 1), dtype=np.float32)


# ---------------------------------------------------------------------------
# process_audio_chunk() — happy path
# ---------------------------------------------------------------------------

class TestProcessAudioChunkHappyPath:

    def test_stt_pipe_called(self, translator):
        translator.config.auto_detect_source_lang = False
        translator.process_audio_chunk(_make_audio())
        translator.stt_pipe.assert_called_once()

    def test_translation_pipe_called(self, translator):
        translator.config.auto_detect_source_lang = False
        translator.process_audio_chunk(_make_audio())
        translator.translator.assert_called_once()

    def test_tts_pipe_called(self, translator):
        translator.config.auto_detect_source_lang = False
        translator.process_audio_chunk(_make_audio())
        translator.tts_pipe.assert_called_once()

    def test_audio_played(self, translator):
        translator.config.auto_detect_source_lang = False
        translator.process_audio_chunk(_make_audio())
        translator.audio_handler.play_audio.assert_called_once()

    def test_history_add_called(self, translator):
        translator.config.auto_detect_source_lang = False
        translator.process_audio_chunk(_make_audio())
        translator.history.add.assert_called_once()

    def test_history_entry_has_source_text(self, translator):
        translator.config.auto_detect_source_lang = False
        translator.process_audio_chunk(_make_audio())
        entry = translator.history.add.call_args[0][0]
        assert entry.source_text == "hello world"

    def test_history_entry_has_translated_text(self, translator):
        translator.config.auto_detect_source_lang = False
        translator.process_audio_chunk(_make_audio())
        entry = translator.history.add.call_args[0][0]
        assert entry.translated_text == "hola mundo"


# ---------------------------------------------------------------------------
# process_audio_chunk() — empty / silent transcription
# ---------------------------------------------------------------------------

class TestProcessAudioChunkEmpty:

    def test_empty_transcription_skips_translation(self, translator):
        translator.config.auto_detect_source_lang = False
        translator.stt_pipe.return_value = {"text": "   "}
        translator.process_audio_chunk(_make_audio())
        translator.translator.assert_not_called()

    def test_empty_transcription_skips_tts(self, translator):
        translator.config.auto_detect_source_lang = False
        translator.stt_pipe.return_value = {"text": ""}
        translator.process_audio_chunk(_make_audio())
        translator.tts_pipe.assert_not_called()

    def test_empty_transcription_skips_history(self, translator):
        translator.config.auto_detect_source_lang = False
        translator.stt_pipe.return_value = {"text": ""}
        translator.process_audio_chunk(_make_audio())
        translator.history.add.assert_not_called()


# ---------------------------------------------------------------------------
# process_audio_chunk() — exception handling
# ---------------------------------------------------------------------------

class TestProcessAudioChunkErrors:

    def test_stt_exception_does_not_propagate(self, translator):
        translator.config.auto_detect_source_lang = False
        translator.stt_pipe.side_effect = RuntimeError("model crash")
        # Must not raise
        translator.process_audio_chunk(_make_audio())

    def test_translation_exception_does_not_propagate(self, translator):
        translator.config.auto_detect_source_lang = False
        translator.translator.side_effect = RuntimeError("translation fail")
        translator.process_audio_chunk(_make_audio())

    def test_tts_exception_does_not_propagate(self, translator):
        translator.config.auto_detect_source_lang = False
        translator.tts_pipe.side_effect = RuntimeError("tts fail")
        translator.process_audio_chunk(_make_audio())

    def test_stt_exception_logged(self, translator, caplog):
        import logging
        translator.config.auto_detect_source_lang = False
        translator.stt_pipe.side_effect = RuntimeError("model crash")
        with caplog.at_level(logging.ERROR, logger="src.translator"):
            translator.process_audio_chunk(_make_audio())
        assert any("model crash" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# play_audio()
# ---------------------------------------------------------------------------

class TestPlayAudio:

    def test_play_audio_delegates(self, translator):
        speech = {"audio": np.zeros(500, dtype=np.float32), "sampling_rate": 16000}
        translator.play_audio(speech)
        translator.audio_handler.play_audio.assert_called_once()

    def test_play_audio_passes_correct_sampling_rate(self, translator):
        speech = {"audio": np.zeros(500, dtype=np.float32), "sampling_rate": 22050}
        translator.play_audio(speech)
        _, sr = translator.audio_handler.play_audio.call_args[0]
        assert sr == 22050

    def test_play_audio_exception_does_not_propagate(self, translator):
        translator.audio_handler.play_audio.side_effect = RuntimeError("speaker gone")
        speech = {"audio": np.zeros(100, dtype=np.float32), "sampling_rate": 16000}
        translator.play_audio(speech)  # must not raise

    def test_play_audio_error_logged(self, translator, caplog):
        import logging
        translator.audio_handler.play_audio.side_effect = RuntimeError("speaker gone")
        speech = {"audio": np.zeros(100, dtype=np.float32), "sampling_rate": 16000}
        with caplog.at_level(logging.ERROR, logger="src.translator"):
            translator.play_audio(speech)
        assert any("speaker gone" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# stop() and is_running
# ---------------------------------------------------------------------------

class TestStartStop:

    def test_stop_sets_is_running_false(self, translator):
        translator.is_running = True
        translator.stop()
        assert translator.is_running is False

    def test_stop_sets_stop_event(self, translator):
        translator._stop_event.clear()
        translator.stop()
        assert translator._stop_event.is_set()


# ---------------------------------------------------------------------------
# _same_language()
# ---------------------------------------------------------------------------

class TestSameLanguage:

    @pytest.mark.parametrize("a,b,expected", [
        ("eng_Latn", "eng_Latn", True),
        ("eng_Latn", "eng_Arab", True),   # same base language prefix "eng"
        ("spa_Latn", "eng_Latn", False),
        ("fra_Latn", "deu_Latn", False),
    ])
    def test_same_language(self, a, b, expected):
        from src.translator import RealTimeTranslator
        assert RealTimeTranslator._same_language(a, b) is expected


# ---------------------------------------------------------------------------
# _set_translation_target()
# ---------------------------------------------------------------------------

class TestSetTranslationTarget:

    def test_sets_forced_bos_token_id(self, translator):
        translator.translator.tokenizer.lang_code_to_id = {"spa_Latn": 99}
        translator.translator.model.config.forced_bos_token_id = None
        translator._set_translation_target("spa_Latn")
        assert translator.translator.model.config.forced_bos_token_id == 99

    def test_unknown_lang_does_not_raise(self, translator):
        translator.translator.tokenizer.lang_code_to_id = {}
        translator._set_translation_target("xyz_Latn")  # key missing → no-op

    def test_attribute_error_does_not_raise(self, translator):
        translator.translator.tokenizer = MagicMock(spec=[])  # no lang_code_to_id
        translator._set_translation_target("spa_Latn")


# ---------------------------------------------------------------------------
# Bidirectional manual alternation
# ---------------------------------------------------------------------------

class TestBidirectionalMode:

    def test_direction_toggles_after_chunk(self, translator):
        translator.config.auto_detect_source_lang = False
        translator.config.bidirectional_mode = True
        translator.tts_pipe_reverse = MagicMock(return_value={
            "audio": np.zeros(100, dtype=np.float32), "sampling_rate": 16000
        })
        translator._active_direction = 0

        translator.process_audio_chunk(_make_audio())
        assert translator._active_direction == 1

        translator.process_audio_chunk(_make_audio())
        assert translator._active_direction == 0

    def test_direction_0_uses_forward_tts(self, translator):
        translator.config.auto_detect_source_lang = False
        translator.config.bidirectional_mode = True
        translator.tts_pipe_reverse = MagicMock(return_value={
            "audio": np.zeros(100, dtype=np.float32), "sampling_rate": 16000
        })
        translator._active_direction = 0
        translator.process_audio_chunk(_make_audio())
        translator.tts_pipe.assert_called_once()
        translator.tts_pipe_reverse.assert_not_called()

    def test_direction_1_uses_reverse_tts(self, translator):
        translator.config.auto_detect_source_lang = False
        translator.config.bidirectional_mode = True
        translator.tts_pipe_reverse = MagicMock(return_value={
            "audio": np.zeros(100, dtype=np.float32), "sampling_rate": 16000
        })
        translator._active_direction = 1
        translator.process_audio_chunk(_make_audio())
        translator.tts_pipe_reverse.assert_called_once()
        translator.tts_pipe.assert_not_called()


# ---------------------------------------------------------------------------
# auto_detect_source_lang=False does not call detect_language
# ---------------------------------------------------------------------------

class TestAutoDetectDisabled:

    @patch("src.translator.detect_language")
    def test_no_detect_when_disabled(self, mock_detect, translator):
        translator.config.auto_detect_source_lang = False
        translator.process_audio_chunk(_make_audio())
        mock_detect.assert_not_called()
