"""
Interactive language selection for the real-time translator.

Shows a numbered menu of common languages and allows the user to pick a
source/target pair at startup.  Also accepts raw NLLB codes for any language
not listed in the menu.

Public API
----------
prompt_language_pair(source_default, target_default, *, stdin, stdout)
    Ask the user to choose source and target languages.
    Returns ``(source_nllb, target_nllb)`` — falls back to the supplied
    defaults on Ctrl-C or when stdin is not a TTY.

is_interactive()
    True when sys.stdin is a real terminal (not redirected / piped).
"""

import sys
from typing import IO, List, Optional, Tuple

from .config import LANGUAGE_CODES

# ── Menu of well-known languages ──────────────────────────────────────────────
# More languages can be appended here; keep the list alphabetically sorted for
# readability.  Any valid NLLB code can be entered directly at the prompt.
_MENU: List[Tuple[str, str]] = sorted(LANGUAGE_CODES.items(), key=lambda x: x[0])

_HEADER = "=" * 56
_DIVIDER = "-" * 56


# ── Public helpers ─────────────────────────────────────────────────────────────

def is_interactive(stream: IO = None) -> bool:
    """Return True when *stream* (default: sys.stdin) is a real TTY."""
    stream = stream or sys.stdin
    try:
        return stream.isatty()
    except AttributeError:
        return False


def list_menu() -> List[Tuple[str, str]]:
    """Return the ordered list of ``(display_name, nllb_code)`` menu entries."""
    return list(_MENU)


def prompt_language_pair(
    source_default: str = "eng_Latn",
    target_default: str = "spa_Latn",
    *,
    stdin: IO = None,
    stdout: IO = None,
) -> Tuple[str, str]:
    """Interactively ask the user to choose source and target language codes.

    Parameters
    ----------
    source_default, target_default:
        Fallback NLLB codes used when stdin is not a TTY, on KeyboardInterrupt,
        or when the user presses Enter without typing anything.
    stdin, stdout:
        Override sys.stdin / sys.stdout (useful for testing).

    Returns
    -------
    (source_nllb, target_nllb) both as NLLB language code strings.
    """
    _in = stdin or sys.stdin
    _out = stdout or sys.stdout

    if not is_interactive(_in):
        return source_default, target_default

    def _write(text: str) -> None:
        _out.write(text)
        _out.flush()

    def _read(prompt: str) -> str:
        _write(prompt)
        return _in.readline().rstrip("\n")

    try:
        _write(f"\n{_HEADER}\n")
        _write(" Language selection\n")
        _write(f"{_HEADER}\n\n")
        _write(" #   Name                       NLLB code\n")
        _write(f"{_DIVIDER}\n")
        for i, (name, code) in enumerate(_MENU, 1):
            _write(f" {i:2d}. {name:<26s} {code}\n")
        _write(f"{_DIVIDER}\n")
        _write(
            " Enter a number, a language name, or any NLLB code directly.\n"
            " Press Enter to keep the default shown in [brackets].\n\n"
        )

        source = _pick_one(
            f" Source language [{source_default}]: ",
            source_default,
            _read,
            _write,
        )
        target = _pick_one(
            f" Target language [{target_default}]: ",
            target_default,
            _read,
            _write,
        )

        src_name = _code_to_name(source)
        tgt_name = _code_to_name(target)
        _write(f"\n{_HEADER}\n")
        _write(f"  Source : {src_name} ({source})\n")
        _write(f"  Target : {tgt_name} ({target})\n")
        _write(f"{_HEADER}\n\n")

        return source, target

    except (KeyboardInterrupt, EOFError):
        _write("\n[Language selection cancelled — using defaults]\n\n")
        return source_default, target_default


# ── Internal helpers ───────────────────────────────────────────────────────────

def _pick_one(prompt: str, default: str, read_fn, write_fn) -> str:
    """Prompt until a valid language code is entered; return the NLLB code."""
    while True:
        raw = read_fn(prompt).strip()
        if not raw:
            return default

        # 1. Numeric menu index
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(_MENU):
                return _MENU[idx][1]
            write_fn(f"  Number must be between 1 and {len(_MENU)}.\n")
            continue
        except ValueError:
            pass

        # 2. Fuzzy name match (case-insensitive substring)
        matches = [
            (name, code)
            for name, code in _MENU
            if raw.lower() in name.lower() or raw.lower() == code.lower()
        ]
        if len(matches) == 1:
            return matches[0][1]
        if len(matches) > 1:
            write_fn(f"  Ambiguous — matches: {', '.join(n for n, _ in matches)}.\n")
            write_fn("  Please be more specific.\n")
            continue

        # 3. Raw NLLB code: accept when it looks like "xxx_Xxxx" (7+ chars with underscore)
        if len(raw) >= 7 and "_" in raw:
            return raw

        write_fn(
            "  Not recognised. Enter a number, a language name, or an NLLB code.\n"
        )


def _code_to_name(code: str) -> str:
    """Return the display name for *code*, or the code itself if not in the menu."""
    for name, c in _MENU:
        if c == code:
            return name
    return code
