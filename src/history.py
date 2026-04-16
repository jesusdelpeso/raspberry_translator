"""
Conversation history — tracks transcript entries and displays them in the
terminal during a live session.

Each entry records:
- wall-clock timestamp
- source language, source text (what was spoken)
- target language, translated text (what was said in output)

Display format (terminal, 80-column-friendly):
  ┌─ 14:03:22  [eng_Latn → spa_Latn] ──────────────────────────────────────────
  │  You  : Hello, how are you?
  │  Trans : Hola, ¿cómo estás?
  └──────────────────────────────────────────────────────────────────────────────
"""

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class TranscriptEntry:
    """One translated utterance."""

    timestamp: str           # ISO-8601 e.g. "2026-04-16T14:03:22.451"
    source_lang: str         # NLLB code, e.g. "eng_Latn"
    target_lang: str         # NLLB code, e.g. "spa_Latn"
    source_text: str         # Transcribed speech
    translated_text: str     # Translation output
    confidence: float = 0.0  # Whisper detection confidence (0 when not detected)

    @staticmethod
    def now() -> str:
        return datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


class ConversationHistory:
    """Keeps a running transcript and renders it to the terminal.

    Parameters
    ----------
    show_history : bool
        When True (default) each entry is printed immediately after it is added.
    save_path : str | None
        When set, the full history is appended to this file as newline-delimited
        JSON after every new entry.
    max_entries : int
        Cap on the in-memory list (oldest entries are dropped when exceeded).
        0 = unlimited.
    """

    _LINE_WIDTH = 78

    def __init__(
        self,
        show_history: bool = True,
        save_path: Optional[str] = None,
        max_entries: int = 0,
    ):
        self.show_history = show_history
        self.save_path = save_path
        self.max_entries = max_entries
        self.entries: List[TranscriptEntry] = []
        self._entry_count = 0  # monotonic count (survives trimming)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, entry: TranscriptEntry) -> None:
        """Append *entry*, optionally render it, optionally persist it."""
        self.entries.append(entry)
        self._entry_count += 1

        if self.max_entries and len(self.entries) > self.max_entries:
            self.entries.pop(0)

        if self.show_history:
            self._print_entry(entry)

        if self.save_path:
            self._append_to_file(entry)

    def clear(self) -> None:
        """Discard all in-memory entries."""
        self.entries.clear()

    def print_summary(self) -> None:
        """Print a full summary of all in-memory entries."""
        if not self.entries:
            print("\n[History] No entries recorded yet.")
            return

        width = self._LINE_WIDTH
        total = self._entry_count
        print(f"\n{'═' * width}")
        print(f"  Session transcript — {total} utterance(s)")
        print(f"{'═' * width}")
        for e in self.entries:
            self._print_entry(e)
        print(f"{'═' * width}\n")

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _print_entry(self, e: TranscriptEntry) -> None:
        """Render one entry as a bordered block to stdout."""
        w = self._LINE_WIDTH
        ts = e.timestamp[11:19]  # just the HH:MM:SS part
        dir_label = f"[{e.source_lang} → {e.target_lang}]"
        header_content = f" {ts}  {dir_label} "
        dashes = "─" * max(0, w - len(header_content) - 1)
        print(f"┌─{header_content}{dashes}")
        # Wrap source text
        for line in self._wrap(f"  {e.source_text}", w - 2):
            print(f"│{line}")
        # Separator
        print(f"│  {'─' * (w - 4)}")
        # Wrap translation
        for line in self._wrap(f"  {e.translated_text}", w - 2):
            print(f"│{line}")
        print(f"└{'─' * (w - 1)}")
        sys.stdout.flush()

    @staticmethod
    def _wrap(text: str, width: int) -> List[str]:
        """Very simple word-wrap that preserves leading indent."""
        indent = len(text) - len(text.lstrip())
        prefix = text[:indent]
        words = text.split()
        lines: List[str] = []
        current = prefix
        for word in words:
            candidate = current + (" " if current.strip() else "") + word
            if len(candidate) <= width:
                current = candidate
            else:
                if current.strip():
                    lines.append(current)
                current = prefix + word
        if current.strip():
            lines.append(current)
        return lines or [prefix]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _append_to_file(self, entry: TranscriptEntry) -> None:
        """Append *entry* as a single JSON line to *save_path*."""
        try:
            path = Path(self.save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"[History] Warning: could not write to {self.save_path}: {exc}")

    def load_from_file(self, path: str) -> None:
        """Load entries from a newline-delimited JSON file produced by this class."""
        p = Path(path)
        if not p.exists():
            return
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                self.entries.append(TranscriptEntry(**data))
        self._entry_count = len(self.entries)
