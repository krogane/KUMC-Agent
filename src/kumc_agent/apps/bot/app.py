from __future__ import annotations

from pathlib import Path

from kumc_agent.apps.agentic import build_agentic_app_context
from kumc_agent.apps.automation import build_automation_app_context
from kumc_agent.apps.foundation import build_foundation_app_context
from kumc_agent.apps.ingestion import build_ingestion_app_context
from kumc_agent.apps.retrieval import build_retrieval_app_context
from kumc_agent.apps.workflow import build_workflow_app_context
from kumc_agent.frontends.discord.app import create_bot
from kumc_agent.utils.logging import configure_logging, default_execution_log_path


def main(*, base_dir: Path | None = None) -> None:
    foundation_context = build_foundation_app_context(base_dir=base_dir)
    configure_logging(
        foundation_context.config.app.log_level,
        file_path=default_execution_log_path(base_dir=foundation_context.config.base_dir),
    )
    token = foundation_context.config.integrations.discord.bot_token
    if not token:
        raise RuntimeError("KUMC_DISCORD_BOT_TOKEN is required to run the bot app.")

    bot = create_bot(
        foundation_context=foundation_context,
        retrieval_context=build_retrieval_app_context(base_dir=base_dir),
        agentic_context=build_agentic_app_context(base_dir=base_dir),
        workflow_context=build_workflow_app_context(base_dir=base_dir),
        automation_context=build_automation_app_context(base_dir=base_dir),
        ingestion_context=build_ingestion_app_context(base_dir=base_dir),
    )
    bot.run(token)
