from __future__ import annotations

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

APP_SRC = Path(__file__).resolve().parent
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

from config import AppConfig, EmbeddingFactory  # noqa: E402
from pipeline.rag_pipeline import RagPipeline  # noqa: E402


def _build_pipeline() -> RagPipeline:
    base_dir = Path(__file__).resolve().parents[2]
    load_dotenv(base_dir / ".env")
    app_config = AppConfig.from_here(base_dir=base_dir)
    embedding_factory = EmbeddingFactory(app_config.embedding_model)
    return RagPipeline(
        index_dir=app_config.index_dir,
        embedding_factory=embedding_factory,
        llm_api_key=app_config.gemini_api_key,
        config=app_config,
    )


def _answer_once(*, pipeline: RagPipeline, query: str) -> None:
    answer = pipeline.answer(query)
    print(answer)


def _interactive(*, pipeline: RagPipeline) -> None:
    print("Enter a query. Type 'exit' or 'quit' to finish.")
    while True:
        try:
            query = input("> ").strip()
        except EOFError:
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        try:
            _answer_once(pipeline=pipeline, query=query)
        except Exception as exc:
            logging.exception("Failed to generate answer")
            print(f"Error: {type(exc).__name__}: {exc}")


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pipeline = _build_pipeline()
    _interactive(pipeline=pipeline)


if __name__ == "__main__":
    main()
