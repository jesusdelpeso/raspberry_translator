"""
Retry utilities for the Real-time Translator.

Usage::

    from .recovery import retry, RetryError

    # Retry up to 3 times with exponential back-off starting at 5 s.
    result = retry(
        some_func,
        max_attempts=3,
        delay_s=5.0,
        label="model load",
    )

    # Retry only for specific exception types.
    result = retry(
        connect_audio,
        max_attempts=5,
        delay_s=2.0,
        exceptions=(OSError, RuntimeError),
        label="audio device",
    )
"""

import logging
import time
from typing import Any, Callable, Optional, Tuple, Type

logger = logging.getLogger(__name__)


class RetryError(RuntimeError):
    """Raised when all retry attempts are exhausted.

    Attributes:
        last_exception: The exception raised on the final attempt.
    """

    def __init__(self, message: str, last_exception: Exception) -> None:
        super().__init__(message)
        self.last_exception = last_exception


def retry(
    fn: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    delay_s: float = 1.0,
    backoff: float = 2.0,
    max_delay_s: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    label: str = "",
    **kwargs: Any,
) -> Any:
    """Call ``fn(*args, **kwargs)`` and retry on failure with exponential back-off.

    Parameters
    ----------
    fn:
        Callable to invoke.
    *args / **kwargs:
        Positional and keyword arguments forwarded to *fn*.
    max_attempts:
        Maximum number of calls, including the first attempt.  Must be >= 1.
    delay_s:
        Seconds to wait before the second attempt.  Subsequent waits are
        multiplied by *backoff* up to *max_delay_s*.
    backoff:
        Multiplier applied to the wait time after each failure.
    max_delay_s:
        Upper bound on the wait time between attempts.
    exceptions:
        Only retry for exceptions that are an instance of one of these types.
        Exceptions not in this tuple are re-raised immediately.
    label:
        Human-readable name for log messages.  Defaults to ``fn.__name__``.

    Returns:
        The return value of ``fn`` on success.

    Raises:
        ValueError: If *max_attempts* < 1.
        RetryError: If all attempts fail for a matching exception.
        Exception: Non-matching exceptions are re-raised without retrying.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    name = label or getattr(fn, "__name__", repr(fn))
    wait = delay_s
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except exceptions as exc:  # type: ignore[misc]
            last_exc = exc
            if attempt == max_attempts:
                break
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1f s...",
                name,
                attempt,
                max_attempts,
                exc,
                wait,
            )
            time.sleep(wait)
            wait = min(wait * backoff, max_delay_s)

    assert last_exc is not None  # guaranteed by the loop
    raise RetryError(
        f"{name} failed after {max_attempts} attempt(s): {last_exc}",
        last_exc,
    )
