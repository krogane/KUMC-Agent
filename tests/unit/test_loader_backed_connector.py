from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.source import BackfillScope
from kumc_agent.infra.connectors.base import LoaderBackedConnector


class AsyncioRunLoader:
    def __init__(self) -> None:
        self.loaded = False

    def load(self) -> int:
        async def _run() -> int:
            self.loaded = True
            return 1

        return asyncio.run(_run())


class LoaderBackedConnectorTests(unittest.TestCase):
    def test_backfill_runs_loader_outside_current_event_loop(self) -> None:
        loader = AsyncioRunLoader()
        connector = LoaderBackedConnector(
            source_kind="dummy",
            loader=loader,
            raw_items=lambda: [],
        )

        async def _collect() -> list[object]:
            return [item async for item in connector.backfill(BackfillScope())]

        items = asyncio.run(_collect())

        self.assertEqual(items, [])
        self.assertTrue(loader.loaded)


if __name__ == "__main__":
    unittest.main()
