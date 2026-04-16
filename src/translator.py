"""
Main translator class orchestrating the translation pipeline
"""

import logging
import threading
import time

import numpy as np

from .audio_handler import AudioHandler
from .config import Config
from .history import ConversationHistory, TranscriptEntry
from .lang_detect import detect_language, whisper_to_nllb
from .models import ModelLoader
from .recovery import RetryError, retry
from .vad import VADDetector

logger = logging.getLogger(__name__)


class RealTimeTranslator:
    """Main translator class handling the full pipeline"""

    def __init__(self, config: Config):
        self.config = config
        self.is_running = False

        # Initialize components
        self.audio_handler = AudioHandler(
            sample_rate=config.sample_rate,
            channels=config.channels,
        )
        self.model_loader = ModelLoader(config)

        # Load VAD model.  Required for streaming mode even when vad_enabled
        # is False, because VADIterator drives the speech-boundary segmentation.
        if config.vad_enabled or config.streaming_enabled:
            self.vad = VADDetector(config)
        else:
            self.vad = None

        # Stop event used to interrupt the streaming generator cleanly.
        self._stop_event = threading.Event()

        # Conversation history
        self.history = ConversationHistory(
            show_history=config.show_history,
            save_path=config.history_save_path,
            max_entries=config.history_max_entries,
        )

        # Active direction in bidirectional mode.
        # 0 = A→B  (source_lang → target_lang)
        # 1 = B→A  (target_lang → source_lang)
        self._active_direction: int = 0

        # Load models — retry on transient failures (download, OOM, etc.)
        try:
            self.stt_pipe = retry(
                self.model_loader.load_stt_model,
                max_attempts=config.model_load_retries,
                delay_s=config.model_load_retry_delay_s,
                label="STT model",
            )
            self.translator = retry(
                self.model_loader.load_translation_model,
                max_attempts=config.model_load_retries,
                delay_s=config.model_load_retry_delay_s,
                label="Translation model",
            )
            self.tts_pipe = retry(
                self.model_loader.load_tts_model,
                max_attempts=config.model_load_retries,
                delay_s=config.model_load_retry_delay_s,
                label="TTS model",
            )
        except RetryError as exc:
            raise RuntimeError(
                f"Failed to load required model after "
                f"{config.model_load_retries} attempt(s): {exc.last_exception}"
            ) from exc.last_exception

        # In bidirectional mode we need a second TTS for the reverse direction
        # (translating *to* source_lang, so TTS speaks source_lang).
        if config.bidirectional_mode:
            logger.info("Bidirectional mode: loading reverse TTS model...")
            try:
                self.tts_pipe_reverse = retry(
                    lambda: self.model_loader.load_tts_for_lang(
                        config.source_lang, fallback=config.tts_model
                    ),
                    max_attempts=config.model_load_retries,
                    delay_s=config.model_load_retry_delay_s,
                    label="Reverse TTS model",
                )
            except RetryError as exc:
                raise RuntimeError(
                    f"Failed to load reverse TTS model after "
                    f"{config.model_load_retries} attempt(s): {exc.last_exception}"
                ) from exc.last_exception
        else:
            self.tts_pipe_reverse = None

        logger.info("All models loaded successfully!")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _same_language(nllb_a: str, nllb_b: str) -> bool:
        """True when two NLLB codes share the same base language (first 3 chars)."""
        return nllb_a[:3].lower() == nllb_b[:3].lower()

    def _set_translation_target(self, tgt_lang: str) -> None:
        """Update the translation pipeline's forced target language token."""
        try:
            tgt_id = self.translator.tokenizer.lang_code_to_id.get(tgt_lang)
            if tgt_id is not None:
                self.translator.model.config.forced_bos_token_id = tgt_id
        except AttributeError:
            pass  # pipeline mocked or doesn't expose these attributes

    # ------------------------------------------------------------------
    # Core chunk processing
    # ------------------------------------------------------------------

    def process_audio_chunk(self, audio_data):
        """
        Process a chunk of audio through the full pipeline.

        Direction routing (A↔B):
        - ``auto_detect_source_lang=True``: Whisper identifies the spoken
          language and the correct translation direction is chosen automatically.
          In bidirectional mode both A→B and B→A are possible in any order.
        - ``auto_detect_source_lang=False`` + ``bidirectional_mode=True``:
          directions alternate automatically after each utterance.
        - Otherwise: standard A→B translation with configured source_lang.
        """
        try:
            audio_float = audio_data.flatten().astype(np.float32)

            # ── Determine translation direction ──────────────────────────────
            src_lang = self.config.source_lang
            tgt_lang = self.config.target_lang
            active_tts = self.tts_pipe  # A→B: TTS speaks target_lang
            confidence = 0.0  # detection confidence (0 when not auto-detected)

            if self.config.auto_detect_source_lang:
                whisper_code, confidence = detect_language(
                    audio_float, self.stt_pipe, self.config.sample_rate
                )
                if whisper_code:
                    try:
                        detected_nllb = whisper_to_nllb(whisper_code)
                        src_lang = detected_nllb
                        logger.debug(
                            "Detected language: %s (%.0f%% confidence) → %s",
                            whisper_code,
                            confidence * 100,
                            detected_nllb,
                        )
                        if self.config.bidirectional_mode:
                            # If detected language matches target_lang → B→A
                            if self._same_language(
                                detected_nllb, self.config.target_lang
                            ):
                                tgt_lang = self.config.source_lang
                                active_tts = self.tts_pipe_reverse
                            # else: A→B defaults apply
                    except ValueError:
                        logger.warning(
                            "Detected language %r has no NLLB mapping; "
                            "using configured source_lang '%s'.",
                            whisper_code,
                            self.config.source_lang,
                        )
                else:
                    logger.debug(
                        "Language detection failed; using configured "
                        "source_lang '%s'.",
                        self.config.source_lang,
                    )
            elif self.config.bidirectional_mode:
                # Manual alternation: direction is toggled after each utterance
                if self._active_direction == 1:
                    src_lang = self.config.target_lang
                    tgt_lang = self.config.source_lang
                    active_tts = self.tts_pipe_reverse

            # ── Label the active direction in bidirectional mode ─────────────
            if self.config.bidirectional_mode:
                logger.debug("Direction: %s → %s", src_lang, tgt_lang)

            # ── Reconfigure translation pipeline ─────────────────────────────
            if src_lang != self.translator.tokenizer.src_lang:
                self.translator.tokenizer.src_lang = src_lang
            self._set_translation_target(tgt_lang)

            # ── Speech-to-Text ────────────────────────────────────────────────
            logger.debug("Transcribing...")
            result = self.stt_pipe(audio_float, sampling_rate=self.config.sample_rate)
            text = result["text"].strip()

            if not text:
                logger.debug("No speech detected")
                return

            logger.info("Detected: %s", text)

            # ── Translation ───────────────────────────────────────────────────
            logger.debug("Translating...")
            translation_result = self.translator(text)
            translated_text = translation_result[0]["translation_text"]
            logger.info("Translated: %s", translated_text)

            # ── Record in conversation history ────────────────────────────────
            self.history.add(
                TranscriptEntry(
                    timestamp=TranscriptEntry.now(),
                    source_lang=src_lang,
                    target_lang=tgt_lang,
                    source_text=text,
                    translated_text=translated_text,
                    confidence=confidence if self.config.auto_detect_source_lang else 0.0,
                )
            )

            # ── Text-to-Speech ────────────────────────────────────────────────
            logger.debug("Generating speech...")
            speech_output = active_tts(translated_text)
            self.play_audio(speech_output)

            # ── Toggle manual direction after success ─────────────────────────
            if self.config.bidirectional_mode and not self.config.auto_detect_source_lang:
                self._active_direction = 1 - self._active_direction

        except Exception as e:
            logger.error("Error processing audio: %s", e)

    def play_audio(self, speech_output):
        """Play the generated audio"""
        try:
            audio_array = speech_output["audio"]
            sampling_rate = speech_output["sampling_rate"]
            self.audio_handler.play_audio(audio_array, sampling_rate)
        except Exception as e:
            logger.error("Error playing audio: %s", e)

    # ------------------------------------------------------------------
    # Listening loops
    # ------------------------------------------------------------------

    def start_listening(self):
        """Start listening to the microphone.

        Routing:
        - Streaming mode (default): segments audio at VAD speech boundaries.
        - Fixed-chunk mode: records fixed-duration chunks.
        - Bidirectional mode prints a mode header; routing is handled inside
          ``process_audio_chunk`` for both streaming and fixed-chunk modes.
        """
        logger.info("Listening... (Press Ctrl+C to stop)")
        if self.config.bidirectional_mode:
            logger.info("Mode: BIDIRECTIONAL conversation")
            logger.info("  Language A: %s", self.config.source_lang)
            logger.info("  Language B: %s", self.config.target_lang)
            if self.config.auto_detect_source_lang:
                logger.info("  Direction: auto-detected from spoken language")
            else:
                logger.info("  Direction: alternating (A→B, then B→A, ...)")
        else:
            logger.info("Source language: %s", self.config.source_lang)
            logger.info("Target language: %s", self.config.target_lang)

        self.is_running = True
        self._stop_event.clear()
        self._active_direction = 0

        if self.config.streaming_enabled and self.vad is not None:
            self._start_streaming()
        else:
            self._start_fixed_chunks()

    def _start_streaming(self):
        """Streaming mode: segment audio at natural speech boundaries.

        If the audio device raises an error mid-stream the method waits and
        attempts to reconnect up to ``config.audio_device_retries`` times with
        exponential back-off before giving up.
        """
        logger.info(
            "Segmentation: streaming (min_silence=%dms, max_duration=%.1fs)",
            self.config.stream_min_silence_ms,
            self.config.stream_max_duration_s,
        )
        max_reconnects = self.config.audio_device_retries
        delay = self.config.audio_device_retry_delay_s
        reconnect = 0

        try:
            while self.is_running:
                try:
                    self._stop_event.clear()
                    for audio_data in self.audio_handler.stream_audio_segments(
                        vad_model=self.vad.model,
                        threshold=self.config.vad_threshold,
                        min_silence_ms=self.config.stream_min_silence_ms,
                        max_duration_s=self.config.stream_max_duration_s,
                        stop_event=self._stop_event,
                    ):
                        if not self.is_running:
                            return
                        self.process_audio_chunk(audio_data)
                        logger.debug("-" * 50)
                        reconnect = 0  # reset on each successful chunk
                    # Generator exhausted cleanly (stop_event was set).
                    return
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # audio device error
                    if not self.is_running:
                        return
                    reconnect += 1
                    if reconnect > max_reconnects:
                        logger.error(
                            "Audio stream failed after %d reconnect attempt(s), "
                            "stopping: %s",
                            max_reconnects,
                            exc,
                        )
                        return
                    logger.warning(
                        "Audio device error (attempt %d/%d): %s "
                        "\u2014 reconnecting in %.1f s...",
                        reconnect,
                        max_reconnects,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2.0, 30.0)
        except KeyboardInterrupt:
            logger.info("Stopping translator...")
        finally:
            self.is_running = False
            self._stop_event.set()
            self.history.print_summary()

    def _start_fixed_chunks(self):
        """Legacy mode: record fixed-duration chunks.

        Audio device errors are retried up to ``config.audio_device_retries``
        times with exponential back-off.  Consecutive errors are counted; a
        successful recording resets the counter.
        """
        logger.info("Segmentation: fixed chunks (%ds)", self.config.recording_duration)
        max_reconnects = self.config.audio_device_retries
        delay = self.config.audio_device_retry_delay_s
        consecutive_errors = 0

        try:
            while self.is_running:
                try:
                    audio_data = self.audio_handler.record_audio(
                        self.config.recording_duration
                    )
                    consecutive_errors = 0
                    delay = self.config.audio_device_retry_delay_s  # reset backoff
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # audio device error
                    consecutive_errors += 1
                    if consecutive_errors > max_reconnects:
                        logger.error(
                            "Audio device failed %d consecutive time(s), stopping.",
                            consecutive_errors,
                        )
                        break
                    logger.warning(
                        "Audio device error (%d/%d): %s "
                        "\u2014 retrying in %.1f s...",
                        consecutive_errors,
                        max_reconnects,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2.0, 30.0)
                    continue

                # Skip silent chunks when VAD is enabled.
                if self.vad is not None and self.config.vad_enabled:
                    if not self.vad.has_speech(audio_data):
                        logger.debug("No speech detected, skipping.")
                        continue

                self.process_audio_chunk(audio_data)
                logger.debug("-" * 50)

        except KeyboardInterrupt:
            logger.info("Stopping translator...")
        finally:
            self.is_running = False
            self.history.print_summary()

    def stop(self):
        """Stop the translator"""
        self.is_running = False
        self._stop_event.set()

