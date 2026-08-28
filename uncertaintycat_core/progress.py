"""Execution-local progress reporting without coupling plugins to a web framework."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

ProgressCallback = Callable[[str, int, str, bool], None]

_progress_callback: ContextVar[ProgressCallback | None] = ContextVar(
    "uncertaintycat_progress_callback", default=None
)


def report_progress(
    phase: str,
    percent: int,
    message: str,
    *,
    indeterminate: bool = False,
) -> None:
    """Publish a bounded phase update when the current execution has a listener."""

    callback = _progress_callback.get()
    if callback is not None:
        callback(phase, max(0, min(100, percent)), message, indeterminate)


@contextmanager
def progress_scope(callback: ProgressCallback | None) -> Iterator[None]:
    """Attach a callback to one analysis execution and restore prior context safely."""

    token = _progress_callback.set(callback)
    try:
        yield
    finally:
        _progress_callback.reset(token)
