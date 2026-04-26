from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.features.rag.components.retrieval import RetrievalComponent


def _chunk(chunk_id: str, updated_at: datetime) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=chunk_id,
        index=0,
        metadata={"updated_at": updated_at.isoformat()},
    )


class RecencyBeforeRrfTests(unittest.TestCase):
    def test_sparse_hits_are_recency_ranked_before_rrf(self) -> None:
        now = datetime.now(timezone.utc)
        old = _chunk("old", now - timedelta(days=365))
        new = _chunk("new", now)

        ranked = RetrievalComponent._apply_recency_to_sparse_hits(
            [(old, 1.0), (new, 1.0)],
            mode="hard",
            recency_weight_soft=0.2,
            recency_weight_hard=0.9,
            recency_half_life_days=30.0,
        )

        self.assertEqual([chunk.id for chunk, _score in ranked], ["new", "old"])


if __name__ == "__main__":
    unittest.main()
