"""
Tests for TODO 9 — Proper logging.

Covers:
  - setup_logging() installs handlers and sets the correct level
  - File handler is added when log_file is specified
  - Log level filtering (DEBUG messages suppressed at INFO level)
  - Invalid level strings fall back to INFO
  - Each src module uses getLogger(__name__), not the root logger directly
  - Config.log_level / Config.log_file defaults and override
  - Config YAML round-trip for logging section
  - CLI --log-level and --log-file args are accepted by parse_arguments()
  - main() calls setup_logging() (no stray print() from non-history modules)
  - ast-based check: no bare print() calls remain outside history / validator
"""

import ast
import io
import logging
import os
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stubs (must come before any src import)
# ---------------------------------------------------------------------------

_sd = types.ModuleType("sounddevice")
_sd.rec = lambda *a, **kw: None
_sd.wait = lambda: None
_sd.play = lambda *a, **kw: None
_sd.InputStream = MagicMock
sys.modules.setdefault("sounddevice", _sd)

_sv = types.ModuleType("silero_vad")
_sv.load_silero_vad = lambda: MagicMock()
_sv.get_speech_timestamps = lambda *a, **kw: []
_sv.VADIterator = MagicMock
sys.modules.setdefault("silero_vad", _sv)

# Stubs for heavy ML libraries
for _mod in ("torch", "transformers", "numpy"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

SRC_DIR = Path(__file__).parent.parent / "src"


def _reset_root_logger():
    """Remove all handlers from the root logger before each test."""
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.setLevel(logging.WARNING)  # neutral starting point


# ---------------------------------------------------------------------------
# setup_logging() tests
# ---------------------------------------------------------------------------

class TestSetupLogging(unittest.TestCase):

    def setUp(self):
        _reset_root_logger()

    def tearDown(self):
        _reset_root_logger()

    def test_installs_stream_handler(self):
        from src.logging_setup import setup_logging
        setup_logging("INFO")
        self.assertTrue(
            any(isinstance(h, logging.StreamHandler) for h in logging.root.handlers),
            "Expected a StreamHandler on the root logger",
        )

    def test_sets_info_level(self):
        from src.logging_setup import setup_logging
        setup_logging("INFO")
        self.assertEqual(logging.root.level, logging.INFO)

    def test_sets_debug_level(self):
        from src.logging_setup import setup_logging
        setup_logging("DEBUG")
        self.assertEqual(logging.root.level, logging.DEBUG)

    def test_sets_warning_level(self):
        from src.logging_setup import setup_logging
        setup_logging("WARNING")
        self.assertEqual(logging.root.level, logging.WARNING)

    def test_sets_error_level(self):
        from src.logging_setup import setup_logging
        setup_logging("ERROR")
        self.assertEqual(logging.root.level, logging.ERROR)

    def test_case_insensitive_level(self):
        from src.logging_setup import setup_logging
        setup_logging("debug")
        self.assertEqual(logging.root.level, logging.DEBUG)

    def test_invalid_level_falls_back_to_info(self):
        from src.logging_setup import setup_logging
        setup_logging("NOTAREALLEVEL")
        self.assertEqual(logging.root.level, logging.INFO)

    def test_file_handler_added(self):
        from src.logging_setup import setup_logging
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            setup_logging("INFO", log_file=log_path)
            file_handlers = [
                h for h in logging.root.handlers
                if isinstance(h, logging.FileHandler)
            ]
            self.assertEqual(len(file_handlers), 1)
            self.assertEqual(
                os.path.abspath(file_handlers[0].baseFilename),
                os.path.abspath(log_path),
            )
        finally:
            for h in logging.root.handlers[:]:
                if isinstance(h, logging.FileHandler):
                    h.close()
            os.unlink(log_path)

    def test_no_file_handler_when_log_file_is_none(self):
        from src.logging_setup import setup_logging
        setup_logging("INFO", log_file=None)
        file_handlers = [
            h for h in logging.root.handlers
            if isinstance(h, logging.FileHandler)
        ]
        self.assertEqual(len(file_handlers), 0)

    def test_log_written_to_file(self):
        from src.logging_setup import setup_logging
        with tempfile.NamedTemporaryFile(
            suffix=".log", delete=False, mode="w"
        ) as f:
            log_path = f.name
        try:
            setup_logging("DEBUG", log_file=log_path)
            logging.getLogger("test.write").debug("hello from test")
            # Flush all handlers
            for h in logging.root.handlers:
                h.flush()
            content = Path(log_path).read_text()
            self.assertIn("hello from test", content)
        finally:
            for h in logging.root.handlers[:]:
                if isinstance(h, logging.FileHandler):
                    h.close()
            os.unlink(log_path)

    def test_info_level_suppresses_debug(self):
        from src.logging_setup import setup_logging
        buf = io.StringIO()
        setup_logging("INFO")
        # Replace stream handler's stream
        for h in logging.root.handlers:
            if isinstance(h, logging.StreamHandler):
                h.stream = buf
        logging.getLogger("test.filter").debug("this should not appear")
        logging.getLogger("test.filter").info("this should appear")
        output = buf.getvalue()
        self.assertNotIn("this should not appear", output)
        self.assertIn("this should appear", output)


# ---------------------------------------------------------------------------
# Module-level logger naming
# ---------------------------------------------------------------------------

class TestModuleLoggers(unittest.TestCase):
    """Each module must use logging.getLogger(__name__), not the root logger."""

    def _get_logger_names(self, source_path: Path) -> list:
        """Parse a module and return the names used in getLogger() calls."""
        tree = ast.parse(source_path.read_text())
        names = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "getLogger"
                and node.args
            ):
                arg = node.args[0]
                if isinstance(arg, ast.Name) and arg.id == "__name__":
                    names.append("__name__")
                elif isinstance(arg, ast.Constant):
                    names.append(arg.value)
        return names

    def _assert_uses_dunder_name(self, module_file: str):
        path = SRC_DIR / module_file
        names = self._get_logger_names(path)
        self.assertTrue(
            len(names) > 0 and all(n == "__name__" for n in names),
            f"{module_file} should use getLogger(__name__), got: {names}",
        )

    def test_translator_logger(self):
        self._assert_uses_dunder_name("translator.py")

    def test_models_logger(self):
        self._assert_uses_dunder_name("models.py")

    def test_audio_handler_logger(self):
        self._assert_uses_dunder_name("audio_handler.py")

    def test_vad_logger(self):
        self._assert_uses_dunder_name("vad.py")

    def test_main_logger(self):
        self._assert_uses_dunder_name("main.py")

    def test_logging_setup_no_module_logger(self):
        """logging_setup.py itself should not declare a module logger."""
        path = SRC_DIR / "logging_setup.py"
        names = self._get_logger_names(path)
        # The setup module configures the root logger — no getLogger(__name__)
        self.assertEqual(names, [], f"Unexpected loggers in logging_setup.py: {names}")


# ---------------------------------------------------------------------------
# No stray print() outside history / validator / main (error msg)
# ---------------------------------------------------------------------------

_ALLOWED_PRINT_FILES = {"history.py", "validator.py"}

# main.py is allowed one deliberate print() for the "Fix the errors" CLI msg
_MAIN_ALLOWED_PRINT_COUNT = 1


class TestNoPrintInModules(unittest.TestCase):

    def _count_print_calls(self, source_path: Path) -> int:
        """Count bare print() calls in a source file using AST."""
        tree = ast.parse(source_path.read_text())
        count = 0
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                count += 1
        return count

    def test_no_print_in_vad(self):
        self.assertEqual(self._count_print_calls(SRC_DIR / "vad.py"), 0)

    def test_no_print_in_models(self):
        self.assertEqual(self._count_print_calls(SRC_DIR / "models.py"), 0)

    def test_no_print_in_audio_handler(self):
        self.assertEqual(self._count_print_calls(SRC_DIR / "audio_handler.py"), 0)

    def test_no_print_in_translator(self):
        self.assertEqual(self._count_print_calls(SRC_DIR / "translator.py"), 0)

    def test_no_print_in_config(self):
        self.assertEqual(self._count_print_calls(SRC_DIR / "config.py"), 0)

    def test_main_has_only_allowed_prints(self):
        count = self._count_print_calls(SRC_DIR / "main.py")
        self.assertLessEqual(
            count,
            _MAIN_ALLOWED_PRINT_COUNT,
            f"main.py has {count} print() calls; expected at most {_MAIN_ALLOWED_PRINT_COUNT}",
        )


# ---------------------------------------------------------------------------
# Config.log_level / Config.log_file fields
# ---------------------------------------------------------------------------

class TestConfigLoggingFields(unittest.TestCase):

    def test_default_log_level(self):
        from src.config import Config
        self.assertEqual(Config().log_level, "INFO")

    def test_default_log_file(self):
        from src.config import Config
        self.assertIsNone(Config().log_file)

    def test_custom_log_level(self):
        from src.config import Config
        c = Config(log_level="DEBUG")
        self.assertEqual(c.log_level, "DEBUG")

    def test_custom_log_file(self):
        from src.config import Config
        c = Config(log_file="/tmp/app.log")
        self.assertEqual(c.log_file, "/tmp/app.log")

    def test_yaml_round_trip_log_level(self):
        from src.config import Config
        c = Config(log_level="WARNING")
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            c.save_yaml(path)
            c2 = Config.from_yaml(path)
            self.assertEqual(c2.log_level, "WARNING")
        finally:
            os.unlink(path)

    def test_yaml_round_trip_log_file(self):
        from src.config import Config
        c = Config(log_file="logs/run.log")
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            c.save_yaml(path)
            c2 = Config.from_yaml(path)
            self.assertEqual(c2.log_file, "logs/run.log")
        finally:
            os.unlink(path)

    def test_yaml_missing_logging_section_uses_defaults(self):
        """YAML without a 'logging' key should use field defaults."""
        import yaml
        from src.config import Config
        minimal = {
            "models": {},
            "languages": {},
            "audio": {},
            "performance": {},
        }
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False, mode="w"
        ) as f:
            yaml.dump(minimal, f)
            path = f.name
        try:
            c = Config.from_yaml(path)
            self.assertEqual(c.log_level, "INFO")
            self.assertIsNone(c.log_file)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLILoggingFlags(unittest.TestCase):

    def _parse(self, argv):
        with patch("sys.argv", ["translator"] + argv):
            from src.main import parse_arguments
            return parse_arguments()

    def test_log_level_default_is_none(self):
        args = self._parse([])
        self.assertIsNone(args.log_level)

    def test_log_level_debug(self):
        args = self._parse(["--log-level", "DEBUG"])
        self.assertEqual(args.log_level, "DEBUG")

    def test_log_level_warning(self):
        args = self._parse(["--log-level", "WARNING"])
        self.assertEqual(args.log_level, "WARNING")

    def test_log_file_default_is_none(self):
        args = self._parse([])
        self.assertIsNone(args.log_file)

    def test_log_file_set(self):
        args = self._parse(["--log-file", "/tmp/t.log"])
        self.assertEqual(args.log_file, "/tmp/t.log")

    def test_invalid_log_level_raises(self):
        with self.assertRaises(SystemExit):
            self._parse(["--log-level", "NOTALEVEL"])


# ---------------------------------------------------------------------------
# main() calls setup_logging()
# ---------------------------------------------------------------------------

class TestMainCallsSetupLogging(unittest.TestCase):

    def test_setup_logging_called_on_startup(self):
        with (
            patch("src.main.setup_logging") as mock_setup,
            patch("src.main.Config.from_yaml_or_default", return_value=MagicMock(
                log_level="INFO", log_file=None,
                source_lang="eng_Latn", target_lang="spa_Latn",
                interactive_lang_select=False,
                bidirectional_mode=False, no_interactive=False,
            )),
            patch("src.main.validate_config") as mock_val,
            patch("src.main.print_validation_report"),
            patch("src.main.is_interactive", return_value=False),
            patch("sys.argv", ["translator", "--validate"]),
        ):
            result = MagicMock()
            result.ok = True
            result.warnings = []
            mock_val.return_value = result

            from src.main import main
            main()

        mock_setup.assert_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
