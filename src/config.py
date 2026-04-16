"""
Configuration settings for the Real-time Translator
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Config:
    """Configuration for the translator"""

    # Model configurations
    stt_model: str = "openai/whisper-small"
    translation_model: str = "facebook/nllb-200-distilled-1.3B"
    tts_model: str = "facebook/mms-tts-eng"
    # When True, the TTS model is derived automatically from target_lang
    # (e.g. spa_Latn → facebook/mms-tts-spa), ignoring the tts_model field.
    # Set to False to use tts_model explicitly.
    tts_auto_select: bool = True

    # Language settings (NLLB language codes)
    source_lang: str = "eng_Latn"
    target_lang: str = "spa_Latn"
    # When True, the spoken language is detected automatically using Whisper's
    # built-in language-identification step, overriding source_lang.
    # source_lang is still used as the fallback if detection fails or the
    # detected language has no NLLB mapping.
    auto_detect_source_lang: bool = True
    # When True and no language was given on the CLI or in a config file, an
    # interactive language-selection menu is shown at startup.  Automatically
    # disabled when stdin is not a TTY (e.g. in scripts or CI).
    interactive_lang_select: bool = True

    # Audio settings
    sample_rate: int = 16000
    recording_duration: int = 5
    channels: int = 1

    # Performance settings
    use_gpu: bool = False
    low_memory: bool = True
    batch_size: int = 16
    max_new_tokens: int = 128
    # When True, ONNX Runtime (via `optimum[onnxruntime]`) is used for model
    # inference instead of PyTorch.  Provides faster CPU inference on Raspberry
    # Pi and other ARM devices without CUDA.  Falls back to PyTorch automatically
    # if `optimum[onnxruntime]` is not installed.
    use_onnx: bool = False

    # Voice Activity Detection settings
    vad_enabled: bool = True
    vad_threshold: float = 0.5

    # Streaming STT settings
    # When True, audio is segmented at natural speech boundaries via
    # VADIterator instead of fixed-duration chunks.  Requires sample_rate
    # to be 8000 or 16000 Hz (Silero constraint).
    streaming_enabled: bool = True
    stream_min_silence_ms: int = 600
    stream_max_duration_s: float = 30.0

    # Bidirectional / conversation mode
    # When True, the translator supports two-way conversation.  Both A→B
    # (source_lang → target_lang) and B→A (target_lang → source_lang)
    # directions are active simultaneously.
    # - With auto_detect_source_lang=True (recommended): Whisper detects the
    #   spoken language each utterance and the correct translation direction is
    #   selected automatically — no explicit turn-taking needed.
    # - With auto_detect_source_lang=False: the system alternates directions
    #   automatically after each utterance (manual turn-taking).
    bidirectional_mode: bool = False

    # Conversation history display
    # show_history: print a bordered transcript block after every utterance.
    # history_save_path: if set, append each entry as a JSON line to that file.
    # history_max_entries: cap in-memory list (0 = unlimited).
    show_history: bool = True
    history_save_path: Optional[str] = None
    history_max_entries: int = 0

    # Logging settings
    # log_level: one of DEBUG, INFO, WARNING, ERROR (case-insensitive).
    # log_file: optional path to append log output (in addition to stderr).
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Error recovery settings
    # model_load_retries: how many times to retry a model download/load failure.
    # model_load_retry_delay_s: initial wait (seconds) before the first retry.
    # audio_device_retries: how many times to attempt to reconnect to the audio
    #   device when it becomes unavailable during a session.
    # audio_device_retry_delay_s: initial wait (seconds) before reconnecting.
    model_load_retries: int = 3
    model_load_retry_delay_s: float = 5.0
    audio_device_retries: int = 5
    audio_device_retry_delay_s: float = 2.0

    def __post_init__(self):
        """Basic type/range validation — runs on every Config construction."""
        if self.recording_duration < 1:
            raise ValueError("Recording duration must be at least 1 second")
        if self.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            raise ValueError(f"Invalid sample rate: {self.sample_rate}")

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Config instance with loaded settings
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
        
        # Extract nested configuration
        models = config_data.get("models", {})
        languages = config_data.get("languages", {})
        audio = config_data.get("audio", {})
        performance = config_data.get("performance", {})
        
        return cls(
            # Models
            stt_model=models.get("stt_model", "openai/whisper-small"),
            translation_model=models.get("translation_model", "facebook/nllb-200-distilled-1.3B"),
            tts_model=models.get("tts_model", "facebook/mms-tts-eng"),
            tts_auto_select=models.get("tts_auto_select", True),
            # Languages
            source_lang=languages.get("source", "eng_Latn"),
            target_lang=languages.get("target", "spa_Latn"),
            auto_detect_source_lang=languages.get("auto_detect_source_lang", True),
            interactive_lang_select=languages.get("interactive_lang_select", True),
            # Audio
            sample_rate=audio.get("sample_rate", 16000),
            recording_duration=audio.get("recording_duration", 5),
            channels=audio.get("channels", 1),
            # Performance
            use_gpu=performance.get("use_gpu", False),
            low_memory=performance.get("low_memory", True),
            batch_size=performance.get("batch_size", 16),
            max_new_tokens=performance.get("max_new_tokens", 128),
            use_onnx=performance.get("use_onnx", False),
            # VAD
            vad_enabled=config_data.get("vad", {}).get("enabled", True),
            vad_threshold=config_data.get("vad", {}).get("threshold", 0.5),
            # Streaming STT
            streaming_enabled=config_data.get("streaming", {}).get("enabled", True),
            stream_min_silence_ms=config_data.get("streaming", {}).get("min_silence_ms", 600),
            stream_max_duration_s=config_data.get("streaming", {}).get("max_duration_s", 30.0),
            # Conversation
            bidirectional_mode=config_data.get("conversation", {}).get("bidirectional", False),
            # History
            show_history=config_data.get("history", {}).get("show", True),
            history_save_path=config_data.get("history", {}).get("save_path", None),
            history_max_entries=config_data.get("history", {}).get("max_entries", 0),
            # Logging
            log_level=config_data.get("logging", {}).get("level", "INFO"),
            log_file=config_data.get("logging", {}).get("file", None),
            # Recovery
            model_load_retries=config_data.get("recovery", {}).get("model_load_retries", 3),
            model_load_retry_delay_s=config_data.get("recovery", {}).get("model_load_retry_delay_s", 5.0),
            audio_device_retries=config_data.get("recovery", {}).get("audio_device_retries", 5),
            audio_device_retry_delay_s=config_data.get("recovery", {}).get("audio_device_retry_delay_s", 2.0),
        )
    
    @classmethod
    def from_yaml_or_default(cls, config_path: Optional[str] = None) -> "Config":
        """
        Load configuration from YAML file or use defaults
        
        Args:
            config_path: Optional path to YAML configuration file
            
        Returns:
            Config instance
        """
        if config_path and Path(config_path).exists():
            return cls.from_yaml(config_path)
        elif config_path:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Config file '%s' not found. Using defaults.", config_path
            )
        
        return cls()
    
    def save_yaml(self, config_path: str):
        """
        Save current configuration to YAML file
        
        Args:
            config_path: Path where to save the configuration
        """
        config_data = {
            "models": {
                "stt_model": self.stt_model,
                "translation_model": self.translation_model,
                "tts_model": self.tts_model,
                "tts_auto_select": self.tts_auto_select,
            },
            "languages": {
                "source": self.source_lang,
                "target": self.target_lang,
                "auto_detect_source_lang": self.auto_detect_source_lang,
                "interactive_lang_select": self.interactive_lang_select,
            },
            "audio": {
                "sample_rate": self.sample_rate,
                "recording_duration": self.recording_duration,
                "channels": self.channels,
            },
            "performance": {
                "use_gpu": self.use_gpu,
                "low_memory": self.low_memory,
                "batch_size": self.batch_size,
                "max_new_tokens": self.max_new_tokens,
                "use_onnx": self.use_onnx,
            },
            "vad": {
                "enabled": self.vad_enabled,
                "threshold": self.vad_threshold,
            },
            "streaming": {
                "enabled": self.streaming_enabled,
                "min_silence_ms": self.stream_min_silence_ms,
                "max_duration_s": self.stream_max_duration_s,
            },
            "conversation": {
                "bidirectional": self.bidirectional_mode,
            },
            "history": {
                "show": self.show_history,
                "save_path": self.history_save_path,
                "max_entries": self.history_max_entries,
            },
            "logging": {
                "level": self.log_level,
                "file": self.log_file,
            },
            "recovery": {
                "model_load_retries": self.model_load_retries,
                "model_load_retry_delay_s": self.model_load_retry_delay_s,
                "audio_device_retries": self.audio_device_retries,
                "audio_device_retry_delay_s": self.audio_device_retry_delay_s,
            },
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        
        import logging as _logging
        _logging.getLogger(__name__).info("Configuration saved to: %s", config_path)


# Common NLLB language codes for reference
LANGUAGE_CODES = {
    "english": "eng_Latn",
    "spanish": "spa_Latn",
    "french": "fra_Latn",
    "german": "deu_Latn",
    "italian": "ita_Latn",
    "portuguese": "por_Latn",
    "russian": "rus_Cyrl",
    "chinese_simplified": "zho_Hans",
    "japanese": "jpn_Jpan",
    "korean": "kor_Hang",
    "arabic": "arb_Arab",
    "hindi": "hin_Deva",
}
