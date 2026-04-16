"""
Tests for ONNX Runtime acceleration support (TODO 14).

Covers:
- Config.use_onnx field defaults and YAML round-trip
- ModelLoader._check_onnx() with optimum present/absent
- load_stt_model() and load_translation_model() route to ORT loaders
- Graceful fallback when optimum is not installed
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call

import pytest

from src.config import Config


# ---------------------------------------------------------------------------
# Config: use_onnx field
# ---------------------------------------------------------------------------

class TestConfigUseOnnxField:
    def test_default_is_false(self):
        c = Config()
        assert c.use_onnx is False

    def test_can_be_set_true(self):
        c = Config(use_onnx=True)
        assert c.use_onnx is True

    def test_yaml_round_trip_false(self, tmp_path):
        c = Config(use_onnx=False)
        path = str(tmp_path / "cfg.yaml")
        c.save_yaml(path)
        loaded = Config.from_yaml(path)
        assert loaded.use_onnx is False

    def test_yaml_round_trip_true(self, tmp_path):
        c = Config(use_onnx=True)
        path = str(tmp_path / "cfg.yaml")
        c.save_yaml(path)
        loaded = Config.from_yaml(path)
        assert loaded.use_onnx is True

    def test_missing_use_onnx_in_yaml_defaults_false(self, tmp_path):
        import yaml
        data = {
            "performance": {
                "use_gpu": False,
                "low_memory": True,
                "batch_size": 8,
                "max_new_tokens": 64,
                # use_onnx intentionally absent
            }
        }
        path = str(tmp_path / "cfg.yaml")
        with open(path, "w") as f:
            yaml.dump(data, f)
        loaded = Config.from_yaml(path)
        assert loaded.use_onnx is False

    def test_use_onnx_persisted_in_save_yaml(self, tmp_path):
        import yaml
        c = Config(use_onnx=True)
        path = str(tmp_path / "cfg.yaml")
        c.save_yaml(path)
        with open(path) as f:
            raw = yaml.safe_load(f)
        assert raw["performance"]["use_onnx"] is True


# ---------------------------------------------------------------------------
# ModelLoader._check_onnx()
# ---------------------------------------------------------------------------

class TestCheckOnnx:
    """_check_onnx() probes for optimum.onnxruntime at import time."""

    def _make_loader(self, use_onnx: bool):
        from src.models import ModelLoader
        cfg = Config(use_onnx=use_onnx)
        with patch("src.models.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.device.return_value = MagicMock(type="cpu")
            mock_torch.float32 = MagicMock()
            loader = ModelLoader.__new__(ModelLoader)
            loader.config = cfg
            loader.use_onnx = use_onnx
            loader.device = MagicMock(type="cpu")
            loader.device_index = -1
            loader.dtype = MagicMock()
        return loader

    def test_returns_true_when_optimum_importable(self):
        loader = self._make_loader(use_onnx=True)
        fake_optimum = types.ModuleType("optimum")
        fake_ort = types.ModuleType("optimum.onnxruntime")
        with patch.dict(sys.modules, {"optimum": fake_optimum, "optimum.onnxruntime": fake_ort}):
            result = loader._check_onnx()
        assert result is True

    def test_returns_false_when_optimum_missing(self):
        loader = self._make_loader(use_onnx=False)
        with patch.dict(sys.modules, {"optimum": None, "optimum.onnxruntime": None}):
            result = loader._check_onnx()
        assert result is False

    def test_logs_warning_when_requested_but_missing(self, caplog):
        import logging
        loader = self._make_loader(use_onnx=True)
        with patch.dict(sys.modules, {"optimum": None, "optimum.onnxruntime": None}):
            with caplog.at_level(logging.WARNING, logger="src.models"):
                loader._check_onnx()
        assert any("optimum[onnxruntime]" in r.message for r in caplog.records)

    def test_no_warning_when_not_requested_and_missing(self, caplog):
        import logging
        loader = self._make_loader(use_onnx=False)
        with patch.dict(sys.modules, {"optimum": None, "optimum.onnxruntime": None}):
            with caplog.at_level(logging.WARNING, logger="src.models"):
                loader._check_onnx()
        assert not any("optimum[onnxruntime]" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# ModelLoader.load_stt_model() — ONNX routing
# ---------------------------------------------------------------------------

class TestLoadSttModelOnnx:
    def _make_loader(self, use_onnx: bool, onnx_available: bool):
        from src.models import ModelLoader
        cfg = Config(use_onnx=use_onnx)
        loader = ModelLoader.__new__(ModelLoader)
        loader.config = cfg
        loader.use_onnx = use_onnx
        loader._onnx_available = onnx_available
        loader.device = MagicMock(type="cpu")
        loader.device_index = -1
        loader.dtype = MagicMock()
        return loader

    def test_calls_ort_stt_when_onnx_enabled_and_available(self):
        loader = self._make_loader(use_onnx=True, onnx_available=True)
        mock_processor = MagicMock()
        with patch("src.models.AutoProcessor") as mock_ap, \
             patch.object(loader, "_load_ort_stt", return_value=MagicMock()) as mock_ort:
            mock_ap.from_pretrained.return_value = mock_processor
            loader.load_stt_model()
        mock_ort.assert_called_once()

    def test_skips_ort_when_onnx_disabled(self):
        loader = self._make_loader(use_onnx=False, onnx_available=True)
        mock_processor = MagicMock()
        with patch("src.models.AutoProcessor") as mock_ap, \
             patch("src.models.AutoModelForSpeechSeq2Seq") as mock_model_cls, \
             patch("src.models.pipeline") as mock_pipe, \
             patch.object(loader, "_load_ort_stt") as mock_ort:
            mock_ap.from_pretrained.return_value = mock_processor
            mock_model_cls.from_pretrained.return_value = MagicMock()
            loader.load_stt_model()
        mock_ort.assert_not_called()

    def test_falls_back_to_pytorch_when_onnx_unavailable(self):
        loader = self._make_loader(use_onnx=True, onnx_available=False)
        mock_processor = MagicMock()
        with patch("src.models.AutoProcessor") as mock_ap, \
             patch("src.models.AutoModelForSpeechSeq2Seq") as mock_model_cls, \
             patch("src.models.pipeline") as mock_pipe, \
             patch.object(loader, "_load_ort_stt") as mock_ort:
            mock_ap.from_pretrained.return_value = mock_processor
            mock_model_cls.from_pretrained.return_value = MagicMock()
            loader.load_stt_model()
        mock_ort.assert_not_called()


# ---------------------------------------------------------------------------
# ModelLoader.load_translation_model() — ONNX routing
# ---------------------------------------------------------------------------

class TestLoadTranslationModelOnnx:
    def _make_loader(self, use_onnx: bool, onnx_available: bool):
        from src.models import ModelLoader
        cfg = Config(use_onnx=use_onnx)
        loader = ModelLoader.__new__(ModelLoader)
        loader.config = cfg
        loader.use_onnx = use_onnx
        loader._onnx_available = onnx_available
        loader.device = MagicMock(type="cpu")
        loader.device_index = -1
        loader.dtype = MagicMock()
        return loader

    def test_calls_ort_translation_when_onnx_enabled_and_available(self):
        loader = self._make_loader(use_onnx=True, onnx_available=True)
        with patch("src.models.AutoTokenizer") as mock_tok, \
             patch.object(loader, "_load_ort_translation", return_value=MagicMock()) as mock_ort:
            mock_tok.from_pretrained.return_value = MagicMock()
            loader.load_translation_model()
        mock_ort.assert_called_once()

    def test_skips_ort_when_onnx_disabled(self):
        loader = self._make_loader(use_onnx=False, onnx_available=True)
        with patch("src.models.AutoTokenizer") as mock_tok, \
             patch("src.models.AutoModelForSeq2SeqLM") as mock_model_cls, \
             patch("src.models.pipeline") as mock_pipe, \
             patch.object(loader, "_load_ort_translation") as mock_ort:
            mock_tok.from_pretrained.return_value = MagicMock()
            mock_model_cls.from_pretrained.return_value = MagicMock()
            loader.load_translation_model()
        mock_ort.assert_not_called()

    def test_falls_back_to_pytorch_when_onnx_unavailable(self):
        loader = self._make_loader(use_onnx=True, onnx_available=False)
        with patch("src.models.AutoTokenizer") as mock_tok, \
             patch("src.models.AutoModelForSeq2SeqLM") as mock_model_cls, \
             patch("src.models.pipeline") as mock_pipe, \
             patch.object(loader, "_load_ort_translation") as mock_ort:
            mock_tok.from_pretrained.return_value = MagicMock()
            mock_model_cls.from_pretrained.return_value = MagicMock()
            loader.load_translation_model()
        mock_ort.assert_not_called()


# ---------------------------------------------------------------------------
# _load_ort_stt / _load_ort_translation internals
# ---------------------------------------------------------------------------

class TestOrtLoaderInternals:
    def _make_loader(self):
        from src.models import ModelLoader
        cfg = Config(use_onnx=True)
        loader = ModelLoader.__new__(ModelLoader)
        loader.config = cfg
        loader.use_onnx = True
        loader._onnx_available = True
        loader.device = MagicMock(type="cpu")
        loader.device_index = -1
        loader.dtype = MagicMock()
        return loader

    def test_ort_stt_imports_ort_model(self):
        loader = self._make_loader()
        mock_model = MagicMock()
        fake_ort_cls = MagicMock(return_value=mock_model)
        fake_ort_mod = MagicMock()
        fake_ort_mod.ORTModelForSpeechSeq2Seq = fake_ort_cls
        mock_processor = MagicMock()

        with patch.dict(sys.modules, {"optimum.onnxruntime": fake_ort_mod}), \
             patch("src.models.pipeline") as mock_pipe:
            loader._load_ort_stt("openai/whisper-small", mock_processor)

        fake_ort_cls.from_pretrained.assert_called_once_with(
            "openai/whisper-small", export=True
        )

    def test_ort_translation_imports_ort_model(self):
        loader = self._make_loader()
        mock_model = MagicMock()
        fake_ort_cls = MagicMock(return_value=mock_model)
        fake_ort_mod = MagicMock()
        fake_ort_mod.ORTModelForSeq2SeqLM = fake_ort_cls
        mock_tokenizer = MagicMock()

        with patch.dict(sys.modules, {"optimum.onnxruntime": fake_ort_mod}), \
             patch("src.models.pipeline") as mock_pipe:
            loader._load_ort_translation("facebook/nllb-200-distilled-1.3B", mock_tokenizer)

        fake_ort_cls.from_pretrained.assert_called_once_with(
            "facebook/nllb-200-distilled-1.3B", export=True
        )

    def test_ort_stt_uses_pipeline(self):
        loader = self._make_loader()
        mock_model = MagicMock()
        fake_ort_mod = MagicMock()
        fake_ort_mod.ORTModelForSpeechSeq2Seq.from_pretrained.return_value = mock_model
        mock_processor = MagicMock()

        with patch.dict(sys.modules, {"optimum.onnxruntime": fake_ort_mod}), \
             patch("src.models.pipeline") as mock_pipe:
            loader._load_ort_stt("openai/whisper-small", mock_processor)

        mock_pipe.assert_called_once()
        call_kwargs = mock_pipe.call_args
        assert call_kwargs[0][0] == "automatic-speech-recognition"

    def test_ort_translation_uses_pipeline_with_lang_codes(self):
        loader = self._make_loader()
        loader.config.source_lang = "eng_Latn"
        loader.config.target_lang = "spa_Latn"
        fake_ort_mod = MagicMock()
        mock_tokenizer = MagicMock()

        with patch.dict(sys.modules, {"optimum.onnxruntime": fake_ort_mod}), \
             patch("src.models.pipeline") as mock_pipe:
            loader._load_ort_translation("facebook/nllb-200-distilled-1.3B", mock_tokenizer)

        _, kwargs = mock_pipe.call_args
        assert kwargs["src_lang"] == "eng_Latn"
        assert kwargs["tgt_lang"] == "spa_Latn"
