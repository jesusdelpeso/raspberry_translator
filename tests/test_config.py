"""
pytest tests for src/config.py.

Covers:
  - Default field values
  - __post_init__ validation (invalid sample_rate, short duration)
  - from_yaml() round-trip for every section
  - from_yaml() with missing optional sections falls back to defaults
  - from_yaml_or_default() with missing file falls back silently
  - CLI-style field overrides after loading
  - LANGUAGE_CODES constant
"""

import os
import tempfile

import pytest
import yaml

from src.config import Config, LANGUAGE_CODES


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:

    def test_stt_model_default(self, default_config):
        assert default_config.stt_model == "openai/whisper-small"

    def test_translation_model_default(self, default_config):
        assert "nllb" in default_config.translation_model.lower()

    def test_tts_model_default(self, default_config):
        assert "mms-tts" in default_config.tts_model

    def test_source_lang_default(self, default_config):
        assert default_config.source_lang == "eng_Latn"

    def test_target_lang_default(self, default_config):
        assert default_config.target_lang == "spa_Latn"

    def test_sample_rate_default(self, default_config):
        assert default_config.sample_rate == 16000

    def test_recording_duration_default(self, default_config):
        assert default_config.recording_duration >= 1

    def test_channels_default(self, default_config):
        assert default_config.channels in (1, 2)

    def test_use_gpu_default(self, default_config):
        assert default_config.use_gpu is False

    def test_vad_enabled_default(self, default_config):
        assert isinstance(default_config.vad_enabled, bool)

    def test_streaming_enabled_default(self, default_config):
        assert isinstance(default_config.streaming_enabled, bool)

    def test_bidirectional_mode_default(self, default_config):
        assert default_config.bidirectional_mode is False

    def test_show_history_default(self, default_config):
        assert default_config.show_history is True

    def test_history_save_path_default(self, default_config):
        assert default_config.history_save_path is None

    def test_log_level_default(self, default_config):
        assert default_config.log_level == "INFO"

    def test_log_file_default(self, default_config):
        assert default_config.log_file is None

    def test_model_load_retries_default(self, default_config):
        assert default_config.model_load_retries >= 1

    def test_audio_device_retries_default(self, default_config):
        assert default_config.audio_device_retries >= 1


# ---------------------------------------------------------------------------
# post_init validation
# ---------------------------------------------------------------------------

class TestConfigValidation:

    def test_invalid_sample_rate_raises(self):
        with pytest.raises(ValueError, match="sample rate"):
            Config(sample_rate=12345)

    def test_recording_duration_too_short_raises(self):
        with pytest.raises(ValueError, match="duration"):
            Config(recording_duration=0)

    @pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000])
    def test_valid_sample_rates(self, rate):
        c = Config(sample_rate=rate)
        assert c.sample_rate == rate


# ---------------------------------------------------------------------------
# Override individual fields
# ---------------------------------------------------------------------------

class TestConfigOverride:

    def test_override_source_lang(self):
        c = Config(source_lang="fra_Latn")
        assert c.source_lang == "fra_Latn"

    def test_override_target_lang(self):
        c = Config(target_lang="deu_Latn")
        assert c.target_lang == "deu_Latn"

    def test_override_duration(self):
        c = Config(recording_duration=10)
        assert c.recording_duration == 10

    def test_override_bidirectional(self):
        c = Config(bidirectional_mode=True)
        assert c.bidirectional_mode is True

    def test_override_log_level(self):
        c = Config(log_level="DEBUG")
        assert c.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_yaml(tmp_path):
    """Return a helper that writes a Config to a temp YAML and reloads it."""
    def _roundtrip(config: Config) -> Config:
        path = str(tmp_path / "config.yaml")
        config.save_yaml(path)
        return Config.from_yaml(path)
    return _roundtrip


class TestConfigYAMLRoundTrip:

    def test_languages(self, tmp_yaml):
        c = Config(source_lang="fra_Latn", target_lang="deu_Latn")
        c2 = tmp_yaml(c)
        assert c2.source_lang == "fra_Latn"
        assert c2.target_lang == "deu_Latn"

    def test_audio_settings(self, tmp_yaml):
        c = Config(sample_rate=8000, recording_duration=3, channels=1)
        c2 = tmp_yaml(c)
        assert c2.sample_rate == 8000
        assert c2.recording_duration == 3

    def test_performance_settings(self, tmp_yaml):
        c = Config(use_gpu=True, low_memory=False, batch_size=4, max_new_tokens=64)
        c2 = tmp_yaml(c)
        assert c2.use_gpu is True
        assert c2.batch_size == 4

    def test_vad_settings(self, tmp_yaml):
        c = Config(vad_enabled=False, vad_threshold=0.3)
        c2 = tmp_yaml(c)
        assert c2.vad_enabled is False
        assert abs(c2.vad_threshold - 0.3) < 1e-6

    def test_streaming_settings(self, tmp_yaml):
        c = Config(streaming_enabled=False, stream_min_silence_ms=400, stream_max_duration_s=15.0)
        c2 = tmp_yaml(c)
        assert c2.streaming_enabled is False
        assert c2.stream_min_silence_ms == 400

    def test_bidirectional(self, tmp_yaml):
        c = Config(bidirectional_mode=True)
        c2 = tmp_yaml(c)
        assert c2.bidirectional_mode is True

    def test_history(self, tmp_yaml):
        c = Config(show_history=False, history_save_path="/tmp/t.jsonl", history_max_entries=50)
        c2 = tmp_yaml(c)
        assert c2.show_history is False
        assert c2.history_save_path == "/tmp/t.jsonl"
        assert c2.history_max_entries == 50

    def test_logging_section(self, tmp_yaml):
        c = Config(log_level="DEBUG", log_file="logs/run.log")
        c2 = tmp_yaml(c)
        assert c2.log_level == "DEBUG"
        assert c2.log_file == "logs/run.log"

    def test_recovery_section(self, tmp_yaml):
        c = Config(model_load_retries=1, model_load_retry_delay_s=0.5,
                   audio_device_retries=2, audio_device_retry_delay_s=1.0)
        c2 = tmp_yaml(c)
        assert c2.model_load_retries == 1
        assert abs(c2.model_load_retry_delay_s - 0.5) < 1e-6
        assert c2.audio_device_retries == 2


# ---------------------------------------------------------------------------
# from_yaml with missing optional sections
# ---------------------------------------------------------------------------

class TestFromYAMLMissingSections:

    def _write_minimal(self, tmp_path, data: dict) -> str:
        path = str(tmp_path / "minimal.yaml")
        with open(path, "w") as f:
            yaml.dump(data, f)
        return path

    def test_missing_vad_section(self, tmp_path):
        path = self._write_minimal(tmp_path, {"models": {}, "languages": {}, "audio": {}, "performance": {}})
        c = Config.from_yaml(path)
        assert isinstance(c.vad_enabled, bool)

    def test_missing_streaming_section(self, tmp_path):
        path = self._write_minimal(tmp_path, {"models": {}, "languages": {}, "audio": {}, "performance": {}})
        c = Config.from_yaml(path)
        assert c.stream_min_silence_ms > 0

    def test_missing_logging_section(self, tmp_path):
        path = self._write_minimal(tmp_path, {"models": {}, "languages": {}, "audio": {}, "performance": {}})
        c = Config.from_yaml(path)
        assert c.log_level == "INFO"
        assert c.log_file is None

    def test_missing_recovery_section(self, tmp_path):
        path = self._write_minimal(tmp_path, {"models": {}, "languages": {}, "audio": {}, "performance": {}})
        c = Config.from_yaml(path)
        assert c.model_load_retries >= 1

    def test_from_yaml_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Config.from_yaml(str(tmp_path / "nonexistent.yaml"))


# ---------------------------------------------------------------------------
# from_yaml_or_default
# ---------------------------------------------------------------------------

class TestFromYAMLOrDefault:

    def test_returns_defaults_for_nonexistent_path(self):
        c = Config.from_yaml_or_default("/nonexistent/path.yaml")
        assert c.source_lang == "eng_Latn"

    def test_returns_defaults_for_none(self):
        c = Config.from_yaml_or_default(None)
        assert isinstance(c, Config)

    def test_loads_file_when_it_exists(self, tmp_path):
        path = str(tmp_path / "c.yaml")
        Config(source_lang="fra_Latn").save_yaml(path)
        c = Config.from_yaml_or_default(path)
        assert c.source_lang == "fra_Latn"


# ---------------------------------------------------------------------------
# LANGUAGE_CODES constant
# ---------------------------------------------------------------------------

class TestLanguageCodes:

    def test_english_entry(self):
        assert "english" in LANGUAGE_CODES
        assert LANGUAGE_CODES["english"] == "eng_Latn"

    def test_all_values_are_nllb_format(self):
        for lang, code in LANGUAGE_CODES.items():
            assert "_" in code, f"Code for {lang!r} missing underscore: {code!r}"
            parts = code.split("_")
            assert len(parts) == 2, f"Unexpected format for {lang!r}: {code!r}"
