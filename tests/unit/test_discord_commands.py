from __future__ import annotations

import asyncio
from types import SimpleNamespace
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.frontends.discord.app import create_bot


def _context(**values: object) -> SimpleNamespace:
    return SimpleNamespace(**values)


class DiscordSlashCommandAdapterTests(unittest.TestCase):
    def test_create_bot_registers_slash_commands(self) -> None:
        config = _context(
            security=_context(
                maintenance_command_author_ids=tuple(),
                discord_guild_allow_list=tuple(),
            )
        )
        bot = create_bot(
            foundation_context=_context(config=config),
            retrieval_context=_context(),
            agentic_context=_context(),
            workflow_context=_context(),
            automation_context=_context(),
            ingestion_context=_context(),
        )
        try:
            command_names = {command.name for command in bot.tree.get_commands()}
            self.assertEqual(
                command_names,
                {"admin", "ask", "work", "approval", "automation"},
            )
        finally:
            asyncio.run(bot.close())


if __name__ == "__main__":
    unittest.main()
