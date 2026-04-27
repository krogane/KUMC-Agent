from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.config.schema import DatabaseSection, MigrationSection
from kumc_agent.infra.database.postgres import PostgresClient
from kumc_agent.infra.migrations.runner import PostgresMigrationRunner


class DatabaseMigrationsTests(unittest.TestCase):
    def test_migration_runner_lists_sql_and_rejects_bad_table_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            migration_dir = Path(tmp)
            (migration_dir / "002_second.sql").write_text("select 2;\n", encoding="utf-8")
            (migration_dir / "001_first.sql").write_text("select 1;\n", encoding="utf-8")
            runner = PostgresMigrationRunner(
                client=PostgresClient(
                    DatabaseSection(url="", connect_timeout_seconds=1.0, application_name="test")
                ),
                config=MigrationSection(
                    directory=migration_dir,
                    table_name="schema_migrations",
                ),
            )
            self.assertEqual(
                [path.name for path in runner.pending_files()],
                ["001_first.sql", "002_second.sql"],
            )

            invalid = PostgresMigrationRunner(
                client=runner.client,
                config=MigrationSection(directory=migration_dir, table_name="bad-name"),
            )
            with self.assertRaises(ValueError):
                invalid._table_name()

    def test_ingestion_migration_contains_tables(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "002_wave2_ingestion.sql"
        ).read_text(encoding="utf-8")
        for table in ("source_items", "documents", "chunks", "secret_findings"):
            self.assertIn(f"create table if not exists {table}", sql)

    def test_retrieval_migration_contains_embedding_and_search_tables(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "003_wave3_retrieval.sql"
        ).read_text(encoding="utf-8")
        for table in ("embeddings", "search_runs", "search_run_results"):
            self.assertIn(f"create table if not exists {table}", sql)

    def test_workflow_migration_contains_tables(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "004_wave4_workflow.sql"
        ).read_text(encoding="utf-8")
        for table in (
            "events",
            "meetings",
            "task_candidates",
            "tasks",
            "schedule_events",
            "approval_records",
        ):
            self.assertIn(f"create table if not exists {table}", sql)

    def test_event_schedule_candidate_migration_tables(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "008_event_schedule_candidates.sql"
        ).read_text(encoding="utf-8")
        for table in ("event_candidates", "schedule_candidates"):
            self.assertIn(f"create table if not exists {table}", sql)

    def test_agentic_docgen_announcement_migration_contains_tables(self) -> None:
        sql = (
            ROOT
            / "infrastructure"
            / "migrations"
            / "005_wave5_agentic_docgen_announcement.sql"
        ).read_text(encoding="utf-8")
        for table in ("agent_runs", "agent_steps", "announcements"):
            self.assertIn(f"create table if not exists {table}", sql)

    def test_minecraft_support_migration_contains_server_operations_table(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "006_wave6_minecraft_support.sql"
        ).read_text(encoding="utf-8")

        self.assertIn("create table if not exists server_operations", sql)
        self.assertIn("dry_run jsonb", sql)

    def test_automation_hardening_migration_contains_tables(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "007_wave7_automation_hardening.sql"
        ).read_text(encoding="utf-8")

        self.assertIn("create table if not exists automation_rules", sql)
        self.assertIn("create table if not exists automation_runs", sql)
        self.assertIn("idempotency_key text not null unique", sql)

    def test_gap_migration_contains_required_tables(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "009_design_gap_tables.sql"
        ).read_text(encoding="utf-8")
        for table in (
            "workflow_candidates",
            "workflow_runs",
            "action_specs",
            "action_runs",
            "action_approvals",
            "llm_calls",
            "tool_calls",
            "indexing_runs",
            "assets",
            "asset_usage_requests",
            "member_profiles",
            "finance_records",
            "eval_sets",
            "eval_cases",
            "eval_runs",
            "eval_results",
            "minecraft_wiki_articles",
        ):
            self.assertIn(f"create table if not exists {table}", sql)


if __name__ == "__main__":
    unittest.main()
