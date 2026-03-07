from __future__ import annotations

import logging
from pathlib import Path


def default_execution_log_path(*, base_dir: Path) -> Path:
    return base_dir / "logs" / "execution.log"


def configure_logging(level: str = "INFO", *, file_path: Path | None = None) -> None:
    resolved = getattr(logging, (level or "INFO").upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    failed_path: Path | None = None
    failed_error: OSError | None = None
    if file_path is not None:
        resolved_path = Path(file_path).expanduser()
        try:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(resolved_path, encoding="utf-8"))
        except OSError as exc:
            failed_path = resolved_path
            failed_error = exc

    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    if failed_path is not None and failed_error is not None:
        logging.getLogger(__name__).warning(
            "Failed to initialize file logging. path=%s error=%s",
            failed_path,
            failed_error,
        )
