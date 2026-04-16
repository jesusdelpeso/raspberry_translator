"""
Model loading and initialization
"""

import logging

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)

from .config import Config
from .tts_lang import nllb_to_mms_model

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and initialization of all AI models"""

    def __init__(self, config: Config):
        self.config = config
        use_cuda = self.config.use_gpu and torch.cuda.is_available()
        if self.config.use_gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but CUDA is not available. Falling back to CPU.")

        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
        self.device_index = 0 if self.device.type == "cuda" else -1
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        logger.info("Using device: %s", self.device.type)

        self.use_onnx = config.use_onnx
        self._onnx_available = self._check_onnx()

    # ------------------------------------------------------------------
    # ONNX Runtime helpers
    # ------------------------------------------------------------------

    def _check_onnx(self) -> bool:
        """Return True if optimum[onnxruntime] is importable.

        Emits a warning when use_onnx=True but the library is missing so the
        user knows they need to ``pip install optimum[onnxruntime]``.
        """
        try:
            import optimum.onnxruntime  # noqa: F401
            logger.info("ONNX Runtime backend is available.")
            return True
        except ImportError:
            if self.use_onnx:
                logger.warning(
                    "ONNX Runtime requested (use_onnx=True) but "
                    "'optimum[onnxruntime]' is not installed. "
                    "Falling back to standard PyTorch inference.\n"
                    "Install with: pip install optimum[onnxruntime]"
                )
            return False

    def _load_ort_stt(self, model_id: str, processor) -> object:
        """Load Whisper via ONNX Runtime using Optimum."""
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

        logger.info("Loading STT model with ONNX Runtime: %s", model_id)
        model = ORTModelForSpeechSeq2Seq.from_pretrained(model_id, export=True)
        return pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=self.config.max_new_tokens,
            chunk_length_s=30,
            batch_size=self.config.batch_size,
            return_timestamps=False,
        )

    def _load_ort_translation(self, model_name: str, tokenizer) -> object:
        """Load NLLB translation model via ONNX Runtime using Optimum."""
        from optimum.onnxruntime import ORTModelForSeq2SeqLM

        logger.info("Loading translation model with ONNX Runtime: %s", model_name)
        model = ORTModelForSeq2SeqLM.from_pretrained(model_name, export=True)
        return pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            src_lang=self.config.source_lang,
            tgt_lang=self.config.target_lang,
            max_length=400,
        )

    # ------------------------------------------------------------------

    def load_stt_model(self):
        """Load Speech-to-Text model (Whisper)"""
        logger.info("Loading Speech-to-Text model...")
        model_id = self.config.stt_model

        processor = AutoProcessor.from_pretrained(model_id)

        if self.use_onnx and self._onnx_available:
            return self._load_ort_stt(model_id, processor)

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=self.config.low_memory,
            use_safetensors=True,
        )
        model.to(self.device)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=self.config.max_new_tokens,
            chunk_length_s=30,
            batch_size=self.config.batch_size,
            return_timestamps=False,
            torch_dtype=self.dtype,
            device=self.device_index,
        )

        return pipe

    def load_translation_model(self):
        """Load NLLB translation model"""
        logger.info("Loading Translation model...")
        model_name = self.config.translation_model

        # Load tokenizer (shared between PyTorch and ORT paths)
        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=self.config.source_lang)

        if self.use_onnx and self._onnx_available:
            return self._load_ort_translation(model_name, tokenizer)

        # Load model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.to(self.device)

        # Create translation pipeline with the loaded model
        translator = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            src_lang=self.config.source_lang,
            tgt_lang=self.config.target_lang,
            max_length=400,
            device=self.device_index,
        )

        return translator

    def load_tts_model(self):
        """Load Text-to-Speech model for the configured target language.

        When ``config.tts_auto_select`` is True (default), the model is derived
        from the target language code (e.g. ``spa_Latn`` → ``facebook/mms-tts-spa``).
        If loading the auto-selected model fails, the method falls back to the
        explicit ``config.tts_model`` value with a warning.

        When ``tts_auto_select`` is False, ``config.tts_model`` is used directly.
        """
        if self.config.tts_auto_select:
            return self.load_tts_for_lang(
                self.config.target_lang, fallback=self.config.tts_model
            )

        model_name = self.config.tts_model
        logger.info("Loading Text-to-Speech model: %s", model_name)
        return pipeline(
            "text-to-speech",
            model=model_name,
            device=self.device_index,
        )

    def load_tts_for_lang(self, nllb_lang: str, fallback: str = "") -> object:
        """Load a TTS pipeline for *any* NLLB language code.

        The model name is derived automatically via :func:`nllb_to_mms_model`.
        If loading fails and *fallback* is non-empty, the fallback model is
        loaded instead; otherwise the exception is re-raised.
        """
        auto_model = nllb_to_mms_model(nllb_lang)
        logger.info(
            "Loading Text-to-Speech model (auto-selected): %s ← language %s",
            auto_model,
            nllb_lang,
        )
        try:
            return pipeline(
                "text-to-speech",
                model=auto_model,
                device=self.device_index,
            )
        except Exception as e:
            if fallback:
                logger.warning(
                    "Could not load TTS model '%s': %s — falling back to: %s",
                    auto_model, e, fallback,
                )
                return pipeline(
                    "text-to-speech",
                    model=fallback,
                    device=self.device_index,
                )
            raise
