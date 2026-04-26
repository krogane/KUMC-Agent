from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.features.rag.access import RagAccessFilter


def _chunk(source_type: str, **metadata: object) -> Chunk:
    return Chunk(
        id="c1",
        document_id="d1",
        text="text",
        index=0,
        metadata={"source_type": source_type, **metadata},
    )


class RagAccessFilterTests(unittest.TestCase):
    def test_drive_and_discord_require_allowed_guild_or_admin_dm(self) -> None:
        access_filter = RagAccessFilter(
            allowed_guild_ids=("100",),
            admin_user_ids=("42",),
        )
        drive = _chunk("docs")
        discord = _chunk("discord_message", guild_id="100")

        self.assertFalse(access_filter.allow_chunk(drive, access=None))
        self.assertFalse(
            access_filter.allow_chunk(
                discord,
                access=AccessContext(user_id="7", guild_id="200"),
            )
        )
        self.assertTrue(
            access_filter.allow_chunk(
                discord,
                access=AccessContext(user_id="7", guild_id="100"),
            )
        )
        self.assertTrue(
            access_filter.allow_chunk(
                drive,
                access=AccessContext(user_id="42", is_admin=True),
            )
        )

    def test_public_sources_pass_but_redaction_deny_always_blocks(self) -> None:
        access_filter = RagAccessFilter(
            allowed_guild_ids=("100",),
            admin_user_ids=("42",),
        )

        self.assertTrue(access_filter.allow_chunk(_chunk("hatenablog"), access=None))
        self.assertFalse(
            access_filter.allow_chunk(
                _chunk("hatenablog", redaction_policy="deny"),
                access=None,
            )
        )
        self.assertFalse(
            access_filter.allow_chunk(
                _chunk("notion", index_status="permission_lost"),
                access=AccessContext(user_id="42", is_admin=True),
            )
        )


if __name__ == "__main__":
    unittest.main()
