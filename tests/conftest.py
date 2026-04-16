"""
Shared pytest fixtures and stubs for the raspberry_translator test suite.

Heavy dependencies (sounddevice, silero_vad, torch, transformers) are stubbed
out here before any src import so every test module benefits automatically.
"""

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Module stubs — installed once at collection time
# ---------------------------------------------------------------------------

def _install_stubs():
    """Register lightweight stubs for unavailable native libraries."""
    if "sounddevice" not in sys.modules:
        sd = types.ModuleType("sounddevice")
        sd.rec = MagicMock(return_value=np.zeros((16000, 1), dtype=np.float32))
        sd.wait = MagicMock()
        sd.play = MagicMock()
        sd.InputStream = MagicMock()
        sd.query_devices = MagicMock(return_value=[
            {"name": "Fake Mic", "max_input_channels": 1, "max_output_channels": 0},
        ])
        sys.modules["sounddevice"] = sd

    if "silero_vad" not in sys.modules:
        sv = types.ModuleType("silero_vad")
        sv.load_silero_vad = MagicMock(return_value=MagicMock())
        sv.get_speech_timestamps = MagicMock(return_value=[])
        sv.VADIterator = MagicMock
        sys.modules["silero_vad"] = sv

    for mod in ("torch", "transformers"):
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()


_install_stubs()


# ---------------------------------------------------------------------------
# Reusable fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_config():
    """A Config instance built with all defaults."""
    from src.config import Config
    return Config()


@pytest.fixture()
def audio_handler():
    """An AudioHandler with 16 kHz / mono settings."""
    from src.audio_handler import AudioHandler
    return AudioHandler(sample_rate=16000, channels=1)


@pytest.fixture()
def mock_stt_pipe():
    """A callable that mimics an ASR pipeline result."""
    pipe = MagicMock()
    pipe.return_value = {"text": "hello world"}
    return pipe


@pytest.fixture()
def mock_translation_pipe():
    """A callable that mimics an NLLB translation pipeline result."""
    pipe = MagicMock()
    pipe.return_value = [{"translation_text": "hola mundo"}]
    pipe.tokenizer = MagicMock()
    pipe.tokenizer.src_lang = "eng_Latn"
    pipe.tokenizer.lang_code_to_id = {"spa_Latn": 1}
    pipe.model = MagicMock()
    pipe.model.config = MagicMock()
    return pipe


@pytest.fixture()
def mock_tts_pipe():
    """A callable that mimics an MMS-TTS pipeline result."""
    pipe = MagicMock()
    pipe.return_value = {
        "audio": np.zeros(16000, dtype=np.float32),
        "sampling_rate": 16000,
    }
    return pipe


@pytest.fixture()
def translator(mock_stt_pipe, mock_translation_pipe, mock_tts_pipe):
    """A RealTimeTranslator with all heavy components mocked out."""
    from unittest.mock import patch

    from src.config import Config
    from src.translator import RealTimeTranslator

    config = Config(
        vad_enabled=False,
        streaming_enabled=False,
        model_load_retries=1,
        model_load_retry_delay_s=0,
        show_history=False,
    )

    with (
        patch("src.translator.ModelLoader") as MockLoader,
        patch("src.translator.AudioHandler"),
        patch("src.translator.ConversationHistory"),
        patch("src.translator.VADDetector"),
    ):
        loader = MagicMock()
        loader.load_stt_model.return_value = mock_stt_pipe
        loader.load_translation_model.return_value = mock_translation_pipe
        loader.load_tts_model.return_value = mock_tts_pipe
        MockLoader.return_value = loader

        t = RealTimeTranslator(config)

    t.history = MagicMock()
    return t
