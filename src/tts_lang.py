"""
Utility for mapping NLLB language codes to Meta MMS TTS model IDs.

NLLB codes use the format ``{iso639_3}_{script}`` (e.g. ``spa_Latn``).
MMS TTS models are published as ``facebook/mms-tts-{iso639_3}``
(e.g. ``facebook/mms-tts-spa``).

In most cases, simply stripping the script suffix gives the correct
MMS code.  A small override table covers the known mismatches.
"""

# Keys are the ISO 639-3 prefixes extracted from NLLB codes.
# Values are the ISO 639-3 codes used by MMS TTS for that language.
_NLLB_TO_MMS_OVERRIDES: dict[str, str] = {
    # Chinese: NLLB uses "zho" (macro-language); MMS uses Mandarin "cmn".
    "zho": "cmn",
    # Modern Standard Arabic: NLLB uses "arb"; MMS uses the broader "ara".
    "arb": "ara",
}

_MMS_TTS_PREFIX = "facebook/mms-tts-"


def nllb_to_mms_model(nllb_lang_code: str) -> str:
    """
    Derive the MMS TTS Hugging Face model ID from an NLLB language code.

    Args:
        nllb_lang_code: NLLB language code, e.g. ``spa_Latn``.

    Returns:
        Hugging Face model ID string, e.g. ``facebook/mms-tts-spa``.

    Raises:
        ValueError: If ``nllb_lang_code`` is empty or has no ``_`` separator.
    """
    if not nllb_lang_code or "_" not in nllb_lang_code:
        raise ValueError(
            f"Invalid NLLB language code: {nllb_lang_code!r}. "
            "Expected format: 'iso639_3_Script' (e.g. 'spa_Latn')."
        )
    iso_code = nllb_lang_code.split("_")[0]
    mms_code = _NLLB_TO_MMS_OVERRIDES.get(iso_code, iso_code)
    return f"{_MMS_TTS_PREFIX}{mms_code}"
