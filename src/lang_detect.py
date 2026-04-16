"""
Whisper language detection and ISO 639-1 → NLLB code mapping.

Whisper speech recognition returns a two- or three-letter ISO 639-1 language
code (e.g. ``"en"``, ``"fr"``, ``"haw"``).  The NLLB translation model uses
codes in the format ``{iso639_3}_{Script}`` (e.g. ``"eng_Latn"``).

``whisper_to_nllb()`` converts between the two.  Unmapped codes raise
``ValueError`` so callers can fall back to the configured source language.

``detect_language()`` runs Whisper's own language-identification step on a
raw audio array without performing full transcription, returning an ISO 639-1
code and confidence score.
"""

from __future__ import annotations

import numpy as np
import torch

# ---------------------------------------------------------------------------
# ISO 639-1 (Whisper) → NLLB code map
# ---------------------------------------------------------------------------
# Covers all 100 languages that Whisper recognises.
# Script tags are chosen to match the default/most-common writing system for
# each language as used by NLLB-200.
# ---------------------------------------------------------------------------
_WHISPER_TO_NLLB: dict[str, str] = {
    "af": "afr_Latn",
    "am": "amh_Ethi",
    "ar": "arb_Arab",
    "as": "asm_Beng",
    "az": "azj_Latn",
    "ba": "bak_Cyrl",
    "be": "bel_Cyrl",
    "bg": "bul_Cyrl",
    "bn": "ben_Beng",
    "bo": "bod_Tibt",
    "br": "bre_Latn",
    "bs": "bos_Latn",
    "ca": "cat_Latn",
    "cs": "ces_Latn",
    "cy": "cym_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "et": "est_Latn",
    "eu": "eus_Latn",
    "fa": "pes_Arab",
    "fi": "fin_Latn",
    "fo": "fao_Latn",
    "fr": "fra_Latn",
    "gl": "glg_Latn",
    "gu": "guj_Gujr",
    "ha": "hau_Latn",
    "haw": "haw_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "ht": "hat_Latn",
    "hu": "hun_Latn",
    "hy": "hye_Armn",
    "id": "ind_Latn",
    "is": "isl_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "jw": "jav_Latn",
    "ka": "kat_Geor",
    "kk": "kaz_Cyrl",
    "km": "khm_Khmr",
    "kn": "kan_Knda",
    "ko": "kor_Hang",
    "la": "lat_Latn",
    "lb": "ltz_Latn",
    "ln": "lin_Latn",
    "lo": "lao_Laoo",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "mg": "plt_Latn",
    "mi": "mri_Latn",
    "mk": "mkd_Cyrl",
    "ml": "mal_Mlym",
    "mn": "khk_Cyrl",
    "mr": "mar_Deva",
    "ms": "zsm_Latn",
    "mt": "mlt_Latn",
    "my": "mya_Mymr",
    "ne": "npi_Deva",
    "nl": "nld_Latn",
    "nn": "nno_Latn",
    "no": "nob_Latn",
    "oc": "oci_Latn",
    "pa": "pan_Guru",
    "pl": "pol_Latn",
    "ps": "pbt_Arab",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sa": "san_Deva",
    "sd": "snd_Arab",
    "si": "sin_Sinh",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sn": "sna_Latn",
    "so": "som_Latn",
    "sq": "als_Latn",
    "sr": "srp_Cyrl",
    "su": "sun_Latn",
    "sv": "swe_Latn",
    "sw": "swh_Latn",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "tg": "tgk_Cyrl",
    "th": "tha_Thai",
    "tk": "tuk_Latn",
    "tl": "tgl_Latn",
    "tr": "tur_Latn",
    "tt": "tat_Cyrl",
    "uk": "ukr_Cyrl",
    "ur": "urd_Arab",
    "uz": "uzn_Latn",
    "vi": "vie_Latn",
    "yi": "ydd_Hebr",
    "yo": "yor_Latn",
    "yue": "yue_Hant",
    "zh": "zho_Hans",
}


def whisper_to_nllb(whisper_lang: str) -> str:
    """
    Convert a Whisper ISO 639-1 language code to an NLLB language code.

    Args:
        whisper_lang: Whisper language code (e.g. ``"en"``, ``"fr"``).

    Returns:
        NLLB language code (e.g. ``"eng_Latn"``).

    Raises:
        ValueError: If the code is not in the mapping table.
    """
    code = whisper_lang.strip().lower()
    if code not in _WHISPER_TO_NLLB:
        raise ValueError(
            f"Whisper language code {whisper_lang!r} has no NLLB mapping. "
            "Use config.source_lang to set the language manually."
        )
    return _WHISPER_TO_NLLB[code]


def detect_language(
    audio_data: np.ndarray,
    stt_pipe,
    sample_rate: int = 16000,
) -> tuple[str, float]:
    """
    Detect the spoken language in *audio_data* using the loaded Whisper model.

    This calls ``model.detect_language()`` directly, which runs only the
    encoder + a small classification head — much cheaper than full transcription.

    Args:
        audio_data: Raw audio as a 1-D float32 numpy array.
        stt_pipe:   The Hugging Face ASR pipeline returned by
                    ``ModelLoader.load_stt_model()``.
        sample_rate: Audio sample rate in Hz.

    Returns:
        A ``(whisper_lang_code, confidence)`` tuple, e.g. ``("en", 0.98)``.
        Returns ``("", 0.0)`` on any error so callers can fall back gracefully.
    """
    try:
        model = stt_pipe.model
        processor = stt_pipe.feature_extractor

        audio_flat = audio_data.flatten().astype(np.float32)

        # Build mel-spectrogram input features.
        inputs = processor(
            audio_flat,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(model.device)

        with torch.no_grad():
            lang_logits = model.detect_language(input_features)

        # detect_language returns a tensor of shape (batch, num_languages).
        lang_probs = lang_logits[0].softmax(dim=-1)
        top_idx = int(lang_probs.argmax())
        confidence = float(lang_probs[top_idx])

        # Resolve the token id → language code via the generation config.
        gen_config = model.generation_config
        # id2label maps token ids to "<|lang|>" strings; strip the angle brackets.
        lang_token = gen_config.lang_to_id  # dict: "<|en|>" -> token_id
        id_to_lang = {v: k.strip("<>|") for k, v in lang_token.items()}
        whisper_lang = id_to_lang.get(top_idx, "")
        return whisper_lang, confidence

    except Exception as e:
        return "", 0.0
