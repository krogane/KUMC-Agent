from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.frontends.discord.commands import (
    parse_command,
    parse_interaction_command,
)


class DiscordCommandParseTests(unittest.TestCase):
    def test_join_command(self) -> None:
        parsed = parse_command(
            content="/ai join",
            prefix="/ai",
            index_command_prefix="/ai build-index",
        )
        self.assertEqual(parsed.kind, "join_vc")

    def test_quit_command(self) -> None:
        parsed = parse_command(
            content="/ai quit",
            prefix="/ai",
            index_command_prefix="/ai build-index",
        )
        self.assertEqual(parsed.kind, "quit_vc")

    def test_chat_command(self) -> None:
        parsed = parse_command(
            content="/ai KUMCとは",
            prefix="/ai",
            index_command_prefix="/ai build-index",
        )
        self.assertEqual(parsed.kind, "chat")
        self.assertEqual(parsed.payload, "KUMCとは")

    def test_chat_fast_command(self) -> None:
        parsed = parse_command(
            content="/ai fast KUMCとは",
            prefix="/ai",
            index_command_prefix="/ai build-index",
        )
        self.assertEqual(parsed.kind, "chat")
        self.assertEqual(parsed.payload, "KUMCとは")
        self.assertTrue(parsed.force_fast_mode)

    def test_build_index_command(self) -> None:
        parsed = parse_command(
            content="/ai build-index",
            prefix="/ai",
            index_command_prefix="/ai build-index",
        )
        self.assertEqual(parsed.kind, "build_index")

    def test_interaction_build_index_subcommand(self) -> None:
        parsed = parse_interaction_command(
            name="ai",
            options=[
                {
                    "name": "build-index",
                    "type": 1,
                    "options": [],
                }
            ],
        )
        self.assertEqual(parsed.kind, "build_index")

    def test_interaction_build_index_action_option(self) -> None:
        parsed = parse_interaction_command(
            name="ai",
            options=[
                {
                    "name": "action",
                    "type": 3,
                    "value": "build-index",
                }
            ],
        )
        self.assertEqual(parsed.kind, "build_index")


if __name__ == "__main__":
    unittest.main()
