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

    # Language settings (NLLB language codes)
    source_lang: str = "eng_Latn"
    target_lang: str = "spa_Latn"

    # Audio settings
    sample_rate: int = 16000
    recording_duration: int = 5
    channels: int = 1

    # Performance settings
    use_gpu: bool = False
    low_memory: bool = True
    batch_size: int = 16
    max_new_tokens: int = 128

    def __post_init__(self):
        """Validate configuration"""
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
            # Languages
            source_lang=languages.get("source", "eng_Latn"),
            target_lang=languages.get("target", "spa_Latn"),
            # Audio
            sample_rate=audio.get("sample_rate", 16000),
            recording_duration=audio.get("recording_duration", 5),
            channels=audio.get("channels", 1),
            # Performance
            use_gpu=performance.get("use_gpu", False),
            low_memory=performance.get("low_memory", True),
            batch_size=performance.get("batch_size", 16),
            max_new_tokens=performance.get("max_new_tokens", 128),
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
            print(f"Warning: Config file '{config_path}' not found. Using defaults.")
        
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
            },
            "languages": {
                "source": self.source_lang,
                "target": self.target_lang,
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
            },
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        
        print(f"Configuration saved to: {config_path}")


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
