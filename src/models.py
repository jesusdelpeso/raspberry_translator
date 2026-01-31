"""
Model loading and initialization
"""

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)

from .config import Config


class ModelLoader:
    """Handles loading and initialization of all AI models"""

    def __init__(self, config: Config):
        self.config = config
        use_cuda = self.config.use_gpu and torch.cuda.is_available()
        if self.config.use_gpu and not torch.cuda.is_available():
            print("Warning: GPU requested but CUDA is not available. Falling back to CPU.")

        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
        self.device_index = 0 if self.device.type == "cuda" else -1
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        print(f"Using device: {self.device.type}")

    def load_stt_model(self):
        """Load Speech-to-Text model (Whisper)"""
        print("Loading Speech-to-Text model...")
        model_id = self.config.stt_model

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=self.config.low_memory,
            use_safetensors=True,
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(model_id)

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
        print("Loading Translation model...")
        model_name = self.config.translation_model
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=self.config.source_lang)
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
        """Load Text-to-Speech model"""
        print("Loading Text-to-Speech model...")
        model_name = self.config.tts_model

        tts_pipe = pipeline(
            "text-to-speech",
            model=model_name,
            device=self.device_index,
        )

        return tts_pipe
