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
        
        # Load tokenizer without specifying src_lang initially
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.to(self.device)

        # Return a wrapper that mimics pipeline behavior
        class TranslationWrapper:
            def __init__(self, model, tokenizer, src_lang, tgt_lang, device):
                self.model = model
                self.tokenizer = tokenizer
                self.src_lang = src_lang
                self.tgt_lang = tgt_lang
                self.device = device
                
            def __call__(self, text):
                # Set source language for tokenizer
                self.tokenizer.src_lang = self.src_lang
                
                # Tokenize input
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=400)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # For NLLB models, language codes need to be converted to token IDs
                # Try different methods to get the token ID
                forced_bos_token_id = None
                
                # Method 1: Try lang_code_to_id (new tokenizers)
                if hasattr(self.tokenizer, 'lang_code_to_id'):
                    forced_bos_token_id = self.tokenizer.lang_code_to_id.get(self.tgt_lang)
                
                # Method 2: Try convert_tokens_to_ids directly
                if forced_bos_token_id is None:
                    forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
                    # If it returns the unknown token ID, it failed
                    if forced_bos_token_id == self.tokenizer.unk_token_id:
                        forced_bos_token_id = None
                
                # Method 3: Try getting from vocab
                if forced_bos_token_id is None and hasattr(self.tokenizer, 'get_vocab'):
                    vocab = self.tokenizer.get_vocab()
                    forced_bos_token_id = vocab.get(self.tgt_lang)
                
                if forced_bos_token_id is None:
                    raise ValueError(
                        f"Could not find token ID for target language '{self.tgt_lang}'. "
                        f"Please check that the language code is valid for this model."
                    )
                
                # Generate translation with optimized parameters
                with torch.no_grad():  # Disable gradient calculation for inference
                    translated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=200,  # Reduced from 400 for faster inference
                        num_beams=1,     # Use greedy decoding (faster than beam search)
                        early_stopping=True,
                        do_sample=False,  # Deterministic output
                    )
                
                # Decode and return in pipeline format
                translation_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                return [{"translation_text": translation_text}]
        
        return TranslationWrapper(model, tokenizer, self.config.source_lang, self.config.target_lang, self.device)


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
