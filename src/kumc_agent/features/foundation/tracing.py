from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4


_TRACE_ID: ContextVar[str] = ContextVar("kumc_trace_id", default="")


def current_trace_id() -> str:
    value = _TRACE_ID.get()
    if value:
        return value
    value = str(uuid4())
    _TRACE_ID.set(value)
    return value


@contextmanager
def trace_scope(trace_id: str | None = None):
    token = _TRACE_ID.set(trace_id or str(uuid4()))
    try:
        yield _TRACE_ID.get()
    finally:
        _TRACE_ID.reset(token)
