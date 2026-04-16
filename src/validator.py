"""
Configuration validation for the Real-time Translator.

``validate_config(config)`` checks all settings and returns a
``ValidationResult`` with structured lists of errors and warnings.

Usage
-----
from src.validator import validate_config, ConfigError

result = validate_config(config)
if not result.ok:
    raise ConfigError(result.format())

Design goals
------------
- No network calls: validation is purely local.
- Warnings for potentially wrong but legal values.
- Errors for values that would definitely cause a runtime crash.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .config import Config

# ---------------------------------------------------------------------------
# NLLB language code set — covers all languages that NLLB-200 supports.
# Derived from the flores200 README.  We include a superset of common codes;
# unknown codes trigger a *warning* (not an error) so users can still try
# exotic codes that may be supported by specific model checkpoints.
# ---------------------------------------------------------------------------
_KNOWN_NLLB_CODES: frozenset[str] = frozenset(
    [
        # A
        "ace_Arab", "ace_Latn", "acm_Arab", "acq_Arab", "aeb_Arab",
        "afr_Latn", "ajp_Arab", "aka_Latn", "amh_Ethi", "apc_Arab",
        "arb_Arab", "arb_Latn", "ars_Arab", "ary_Arab", "arz_Arab",
        "asm_Beng", "ast_Latn", "awa_Deva", "ayr_Latn", "azb_Arab",
        "azj_Latn",
        # B
        "bak_Cyrl", "bam_Latn", "ban_Latn", "bel_Cyrl", "bem_Latn",
        "ben_Beng", "bho_Deva", "bjn_Arab", "bjn_Latn", "bod_Tibt",
        "bos_Latn", "bug_Latn", "bul_Cyrl",
        # C
        "cat_Latn", "ceb_Latn", "ces_Latn", "cjk_Latn", "ckb_Arab",
        "crh_Latn", "cym_Latn",
        # D
        "dan_Latn", "deu_Latn", "dik_Latn", "dyu_Latn", "dzo_Tibt",
        # E
        "ell_Grek", "eng_Latn", "epo_Latn", "est_Latn", "eus_Latn",
        "ewe_Latn",
        # F
        "fao_Latn", "fij_Latn", "fin_Latn", "fon_Latn", "fra_Latn",
        "fur_Latn", "fuv_Latn",
        # G
        "gla_Latn", "gle_Latn", "glg_Latn", "grn_Latn", "guj_Gujr",
        # H
        "hat_Latn", "hau_Latn", "haw_Latn", "heb_Hebr", "hin_Deva",
        "hne_Deva", "hrv_Latn", "hun_Latn", "hye_Armn",
        # I
        "ibo_Latn", "ilo_Latn", "ind_Latn", "isl_Latn", "ita_Latn",
        # J
        "jav_Latn", "jpn_Jpan",
        # K
        "kab_Latn", "kac_Latn", "kam_Latn", "kan_Knda", "kas_Arab",
        "kas_Deva", "kat_Geor", "kaz_Cyrl", "kbp_Latn", "kea_Latn",
        "khk_Cyrl", "khm_Khmr", "kik_Latn", "kin_Latn", "kir_Cyrl",
        "kmb_Latn", "kmr_Latn", "knc_Arab", "knc_Latn", "kon_Latn",
        "kor_Hang", "ktu_Latn", "kur_Latn",
        # L
        "lao_Laoo", "lat_Latn", "lij_Latn", "lim_Latn", "lin_Latn",
        "lit_Latn", "lmo_Latn", "ltg_Latn", "ltz_Latn", "lua_Latn",
        "lug_Latn", "luo_Latn", "lus_Latn", "lvs_Latn",
        # M
        "mag_Deva", "mai_Deva", "mal_Mlym", "mar_Deva", "min_Arab",
        "min_Latn", "mkd_Cyrl", "mlt_Latn", "mni_Beng", "mos_Latn",
        "mri_Latn", "mya_Mymr",
        # N
        "nld_Latn", "nno_Latn", "nob_Latn", "npi_Deva", "nso_Latn",
        "nus_Latn", "nya_Latn",
        # O
        "oci_Latn", "ory_Orya",
        # P
        "pag_Latn", "pan_Guru", "pap_Latn", "pbt_Arab", "pes_Arab",
        "plt_Latn", "pol_Latn", "por_Latn",
        # R
        "ron_Latn", "run_Latn", "rus_Cyrl",
        # S
        "sag_Latn", "san_Deva", "sat_Olck", "scn_Latn", "shn_Mymr",
        "sin_Sinh", "slk_Latn", "slv_Latn", "smo_Latn", "sna_Latn",
        "snd_Arab", "som_Latn", "sot_Latn", "spa_Latn", "srd_Latn",
        "srp_Cyrl", "ssw_Latn", "sun_Latn", "swe_Latn", "swh_Latn",
        # T
        "szl_Latn", "tam_Taml", "taq_Latn", "taq_Tfng", "tat_Cyrl",
        "tel_Telu", "tgk_Cyrl", "tgl_Latn", "tha_Thai", "tir_Ethi",
        "tpi_Latn", "tsn_Latn", "tso_Latn", "tuk_Latn", "tum_Latn",
        "tur_Latn", "twi_Latn",
        # U
        "tzm_Tfng", "uig_Arab", "ukr_Cyrl", "umb_Latn", "urd_Arab",
        "uzn_Latn",
        # V
        "vec_Latn", "vie_Latn",
        # W
        "war_Latn", "wol_Latn",
        # X
        "xho_Latn",
        # Y
        "ydd_Hebr", "yor_Latn", "yue_Hant",
        # Z
        "zho_Hans", "zho_Hant", "zsm_Latn", "zul_Latn",
    ]
)

# NLLB code pattern: exactly "xxx_Xxxx" where prefix is 2-4 lower-alpha chars
# and suffix is 4 ASCII chars starting with an uppercase letter.
_NLLB_PATTERN = re.compile(r"^[a-z]{2,4}_[A-Z][a-zA-Z]{3}$")

# Whisper model identifiers (local paths are fine too)
_VALID_WHISPER_MODELS = frozenset(
    [
        "openai/whisper-tiny",
        "openai/whisper-tiny.en",
        "openai/whisper-base",
        "openai/whisper-base.en",
        "openai/whisper-small",
        "openai/whisper-small.en",
        "openai/whisper-medium",
        "openai/whisper-medium.en",
        "openai/whisper-large",
        "openai/whisper-large-v2",
        "openai/whisper-large-v3",
    ]
)

_VALID_NLLB_MODELS = frozenset(
    [
        "facebook/nllb-200-distilled-600M",
        "facebook/nllb-200-distilled-1.3B",
        "facebook/nllb-200-1.3B",
        "facebook/nllb-200-3.3B",
    ]
)

_VALID_SAMPLE_RATES = frozenset([8000, 16000, 22050, 44100, 48000])
_VAD_SAMPLE_RATES = frozenset([8000, 16000])


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Holds errors and warnings from a config validation run."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when there are no errors (warnings are allowed)."""
        return len(self.errors) == 0

    def format(self) -> str:
        """Return a human-readable summary."""
        lines: List[str] = []
        if self.errors:
            lines.append(f"Configuration errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"  ✗ {e}")
        if self.warnings:
            lines.append(f"Configuration warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        return "\n".join(lines) if lines else "Configuration OK."


class ConfigError(ValueError):
    """Raised when configuration validation fails."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_config(config: "Config") -> ValidationResult:
    """Validate all settings in *config* and return a ``ValidationResult``.

    Errors are emitted for values that will definitely crash at runtime.
    Warnings are emitted for values that are unusual or potentially wrong
    but may still work (e.g. an exotic NLLB code not in our known set).
    """
    result = ValidationResult()
    _check_languages(config, result)
    _check_models(config, result)
    _check_audio(config, result)
    _check_performance(config, result)
    _check_vad_streaming(config, result)
    _check_history(config, result)
    return result


def print_validation_report(result: ValidationResult) -> None:
    """Print the validation result to stdout, suitable for CLI output."""
    if result.ok and not result.warnings:
        print("✓ Configuration is valid.")
        return
    print(result.format())


# ---------------------------------------------------------------------------
# Internal checks
# ---------------------------------------------------------------------------


def _check_languages(config: "Config", result: ValidationResult) -> None:
    for attr, label in [("source_lang", "source_lang"), ("target_lang", "target_lang")]:
        code = getattr(config, attr)
        if not _NLLB_PATTERN.match(code):
            result.errors.append(
                f"{label} '{code}' is not a valid NLLB code format. "
                f"Expected pattern: 'xxx_Xxxx' (e.g. 'eng_Latn')."
            )
        elif code not in _KNOWN_NLLB_CODES:
            result.warnings.append(
                f"{label} '{code}' is not in the known NLLB language list. "
                f"Double-check the code — it may still work with some model variants."
            )

    if config.source_lang == config.target_lang:
        result.warnings.append(
            "source_lang and target_lang are identical — translation output "
            "will be the same as the input."
        )

    if config.vad_threshold < 0.0 or config.vad_threshold > 1.0:
        result.errors.append(
            f"vad_threshold must be between 0.0 and 1.0, got {config.vad_threshold}."
        )


def _check_models(config: "Config", result: ValidationResult) -> None:
    stt = config.stt_model
    if not stt:
        result.errors.append("stt_model is empty.")
    elif stt not in _VALID_WHISPER_MODELS and not _looks_like_local_path(stt):
        result.warnings.append(
            f"stt_model '{stt}' is not a recognised Whisper model identifier. "
            f"Valid options: {', '.join(sorted(_VALID_WHISPER_MODELS))}. "
            f"A local path is also accepted."
        )

    tm = config.translation_model
    if not tm:
        result.errors.append("translation_model is empty.")
    elif tm not in _VALID_NLLB_MODELS and not _looks_like_local_path(tm):
        result.warnings.append(
            f"translation_model '{tm}' is not a recognised NLLB model identifier. "
            f"Valid options: {', '.join(sorted(_VALID_NLLB_MODELS))}."
        )

    if not config.tts_auto_select:
        tts = config.tts_model
        if not tts:
            result.errors.append("tts_model is empty and tts_auto_select is False.")
        elif not tts.startswith("facebook/mms-tts-") and not _looks_like_local_path(tts):
            result.warnings.append(
                f"tts_model '{tts}' does not follow the expected 'facebook/mms-tts-*' "
                f"pattern. Ensure it is a valid HuggingFace TTS model."
            )


def _check_audio(config: "Config", result: ValidationResult) -> None:
    if config.sample_rate not in _VALID_SAMPLE_RATES:
        result.errors.append(
            f"sample_rate {config.sample_rate} is not valid. "
            f"Choose from: {sorted(_VALID_SAMPLE_RATES)}."
        )
    if config.recording_duration < 1:
        result.errors.append(
            f"recording_duration must be ≥ 1 second, got {config.recording_duration}."
        )
    if config.channels not in (1, 2):
        result.errors.append(
            f"channels must be 1 (mono) or 2 (stereo), got {config.channels}."
        )


def _check_performance(config: "Config", result: ValidationResult) -> None:
    if config.batch_size < 1:
        result.errors.append(
            f"batch_size must be ≥ 1, got {config.batch_size}."
        )
    if config.max_new_tokens < 1:
        result.errors.append(
            f"max_new_tokens must be ≥ 1, got {config.max_new_tokens}."
        )


def _check_vad_streaming(config: "Config", result: ValidationResult) -> None:
    if (config.vad_enabled or config.streaming_enabled) and \
            config.sample_rate not in _VAD_SAMPLE_RATES:
        result.errors.append(
            f"VAD / streaming requires sample_rate of 8000 or 16000 Hz, "
            f"but sample_rate is {config.sample_rate}. "
            f"Set sample_rate to 8000 or 16000, or disable both vad_enabled "
            f"and streaming_enabled."
        )
    if config.stream_min_silence_ms < 100:
        result.warnings.append(
            f"stream_min_silence_ms is {config.stream_min_silence_ms} ms — "
            f"very short silences may cause excessive segmentation."
        )
    if config.stream_max_duration_s < 1.0:
        result.errors.append(
            f"stream_max_duration_s must be ≥ 1.0 s, got {config.stream_max_duration_s}."
        )


def _check_history(config: "Config", result: ValidationResult) -> None:
    if config.history_max_entries < 0:
        result.errors.append(
            f"history_max_entries must be ≥ 0 (0 = unlimited), "
            f"got {config.history_max_entries}."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_like_local_path(s: str) -> bool:
    """True when *s* looks like a filesystem path rather than a HF model ID."""
    return s.startswith("/") or s.startswith("./") or s.startswith("../") or "\\" in s
