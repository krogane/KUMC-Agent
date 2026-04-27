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
            ROOT / "infrastructure" / "migrations" / "002_ingestion_sources_documents.sql"
        ).read_text(encoding="utf-8")
        for table in ("source_items", "documents", "chunks", "secret_findings"):
            self.assertIn(f"create table if not exists {table}", sql)

    def test_retrieval_migration_contains_embedding_and_search_tables(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "003_retrieval_embeddings_search.sql"
        ).read_text(encoding="utf-8")
        for table in ("embeddings", "search_runs", "search_run_results"):
            self.assertIn(f"create table if not exists {table}", sql)

    def test_workflow_migration_contains_tables(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "004_workflow_events_tasks_approvals.sql"
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
            ROOT
            / "infrastructure"
            / "migrations"
            / "008_workflow_event_schedule_candidates.sql"
        ).read_text(encoding="utf-8")
        for table in ("event_candidates", "schedule_candidates"):
            self.assertIn(f"create table if not exists {table}", sql)

    def test_agentic_docgen_announcement_migration_contains_tables(self) -> None:
        sql = (
            ROOT
            / "infrastructure"
            / "migrations"
            / "005_agentic_runs_announcements.sql"
        ).read_text(encoding="utf-8")
        for table in ("agent_runs", "agent_steps", "announcements"):
            self.assertIn(f"create table if not exists {table}", sql)

    def test_minecraft_support_migration_contains_server_operations_table(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "006_minecraft_server_operations.sql"
        ).read_text(encoding="utf-8")

        self.assertIn("create table if not exists server_operations", sql)
        self.assertIn("dry_run jsonb", sql)

    def test_automation_hardening_migration_contains_tables(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "007_automation_rules_runs.sql"
        ).read_text(encoding="utf-8")

        self.assertIn("create table if not exists automation_rules", sql)
        self.assertIn("create table if not exists automation_runs", sql)
        self.assertIn("idempotency_key text not null unique", sql)

    def test_workflow_action_execution_migration_contains_required_tables(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "009_workflow_action_execution.sql"
        ).read_text(encoding="utf-8")
        for table in (
            "workflow_candidates",
            "workflow_runs",
            "action_specs",
            "action_runs",
            "action_approvals",
        ):
            self.assertIn(f"create table if not exists {table}", sql)
        self.assertIn("create unique index if not exists uq_action_runs_idempotency_key", sql)

    def test_observability_migration_contains_required_tables(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "010_observability_llm_tool_calls.sql"
        ).read_text(encoding="utf-8")
        for table in ("llm_calls", "tool_calls"):
            self.assertIn(f"create table if not exists {table}", sql)

    def test_indexing_assets_migration_contains_required_tables(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "011_ingestion_indexing_assets.sql"
        ).read_text(encoding="utf-8")
        for table in (
            "indexing_runs",
            "assets",
        ):
            self.assertIn(f"create table if not exists {table}", sql)

    def test_member_profiles_migration_contains_required_tables(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "012_member_profiles.sql"
        ).read_text(encoding="utf-8")
        for table in (
            "member_profiles",
        ):
            self.assertIn(f"create table if not exists {table}", sql)

    def test_finance_migration_contains_required_tables(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "013_finance_records.sql"
        ).read_text(encoding="utf-8")
        for table in (
            "finance_records",
        ):
            self.assertIn(f"create table if not exists {table}", sql)

    def test_evaluations_migration_contains_required_tables(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "014_evaluations.sql"
        ).read_text(encoding="utf-8")
        for table in (
            "eval_sets",
            "eval_cases",
            "eval_runs",
            "eval_results",
        ):
            self.assertIn(f"create table if not exists {table}", sql)

    def test_minecraft_wiki_migration_contains_required_tables(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "015_minecraft_wiki_articles.sql"
        ).read_text(encoding="utf-8")
        for table in (
            "minecraft_wiki_articles",
        ):
            self.assertIn(f"create table if not exists {table}", sql)

    def test_task_management_hardening_migration_contains_required_tables(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "016_task_management_hardening.sql"
        ).read_text(encoding="utf-8")
        for table in (
            "task_change_candidates",
            "task_approval_batches",
        ):
            self.assertIn(f"create table if not exists {table}", sql)
        self.assertIn("idx_tasks_assignee_due", sql)


if __name__ == "__main__":
    unittest.main()
