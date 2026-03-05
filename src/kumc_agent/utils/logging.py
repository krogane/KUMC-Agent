from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    resolved = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
