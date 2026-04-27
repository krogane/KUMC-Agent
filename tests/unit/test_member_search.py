from __future__ import annotations

from dataclasses import replace
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.operations import MemberProfile
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.domain.models.workflow import WorkRequest
from kumc_agent.features.member_search import (
    DiscordMemberRecord,
    MemberProfileBuildService,
    MemberSearchConfig,
    MemberSearchService,
)
from kumc_agent.features.member_search.service import (
    MemberProfileGenerator,
    MemberProfileIndexService,
    extract_conditions,
    mask_sensitive_text,
)
from kumc_agent.features.workflow import WorkflowService
from kumc_agent.infra.operations import FileOperationsRepository
from kumc_agent.infra.workflow import FileWorkflowRepository


class _Embedder:
    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vector(text) for text in texts]) if texts else np.empty((0, 4))

    def _vector(self, text: str) -> np.ndarray:
        lowered = text.lower()
        return np.asarray(
            [
                1.0 if "design" in lowered or "デザイン" in lowered else 0.0,
                1.0 if "event" in lowered or "イベント" in lowered else 0.0,
                1.0 if "video" in lowered or "動画" in lowered else 0.0,
                1.0,
            ],
            dtype=np.float32,
        )


class _FailingEmbedder(_Embedder):
    def embed_query(self, text: str) -> np.ndarray:
        raise RuntimeError("dense unavailable")


class _Directory:
    def __init__(self, members: list[DiscordMemberRecord]) -> None:
        self.members = members

    def list_members(self, *, guild_id: str) -> list[DiscordMemberRecord]:
        return [member for member in self.members if member.guild_id == guild_id]


class _EvidenceSource:
    def collect(self, *, member: DiscordMemberRecord, access: AccessContext) -> tuple[dict[str, object], ...]:
        return (
            {
                "source_type": "docs",
                "source_item_id": "doc-1",
                "chunk_id": "chunk-1",
                "label": "企画メモ",
                "quote": f"{member.display_name} はイベント告知デザインを担当",
                "access_scope": {"guild_ids": [member.guild_id]},
            },
        )


class _LLM:
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        return json.dumps(
            {
                "skills": ["イベント告知デザイン"],
                "interests": ["動画"],
                "past_assignments": ["新歓告知"],
                "confidence": "medium",
            },
            ensure_ascii=False,
        )


class MemberSearchTests(unittest.TestCase):
    def _repo(self, root: Path) -> FileOperationsRepository:
        return FileOperationsRepository(root_dir=root / "operations")

    def _config(self) -> MemberSearchConfig:
        return MemberSearchConfig(
            allowed_guild_ids=("g1",),
            admin_user_ids=("admin-1",),
            search_limit=3,
            dense_enabled=True,
        )

    def _profiles(self) -> tuple[MemberProfile, MemberProfile]:
        return (
            MemberProfile(
                id="p1",
                display_name="Alice",
                discord_user_id="111111",
                roles=("designer", "event"),
                skills=("イベント告知デザイン",),
                interests=("動画",),
                past_assignments=("新歓告知",),
                evidence=(
                    {
                        "source_type": "docs",
                        "source_item_id": "doc-1",
                        "chunk_id": "c1",
                        "label": "新歓メモ",
                        "quote": "Alice がイベント告知デザインを担当",
                        "access_scope": {"guild_ids": ["g1"]},
                    },
                    {
                        "source_type": "docs",
                        "source_item_id": "secret-doc",
                        "chunk_id": "c2",
                        "label": "管理メモ",
                        "quote": "管理者だけの根拠",
                        "access_scope": {"admin_only": True},
                    },
                ),
                access_scope={"guild_ids": ["g1"]},
                metadata={"profile_status": "generated", "role_ids": ["r-design"]},
            ),
            MemberProfile(
                id="p2",
                display_name="Bob",
                discord_user_id="222222",
                roles=("server",),
                skills=("サーバー運用",),
                access_scope={"guild_ids": ["g1"]},
                metadata={"profile_status": "generated"},
            ),
        )

    def test_repository_saves_evidence_and_reads_previous_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._repo(Path(tmp))
            profile = self._profiles()[0]
            repo.save_member_profile(profile)
            path = Path(tmp) / "operations" / "member_profiles.jsonl"
            with path.open("a", encoding="utf-8") as fw:
                fw.write(
                    json.dumps(
                        {
                            "id": "old-payload",
                            "display_name": "Previous",
                            "discord_user_id": "333333",
                            "roles": [],
                            "skills": [],
                            "interests": [],
                            "past_assignments": [],
                            "access_scope": {},
                            "metadata": {},
                        }
                    )
                    + "\n"
                )

            profiles = {item.id: item for item in repo.list_member_profiles()}

            self.assertEqual(profiles["p1"].evidence[0]["chunk_id"], "c1")
            self.assertEqual(profiles["old-payload"].evidence, tuple())

    def test_extract_conditions(self) -> None:
        conditions = extract_conditions("<@111111> role:designer @Alice <@&999999>")

        self.assertEqual(conditions.user_ids, ("111111",))
        self.assertIn("Alice", conditions.display_names)
        self.assertEqual(conditions.role_ids, ("999999",))
        self.assertIn("designer", conditions.role_names)

    def test_search_authorization_and_access_filtered_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._repo(Path(tmp))
            for profile in self._profiles():
                repo.save_member_profile(profile)
            service = MemberSearchService(
                repository=repo,
                config=self._config(),
                embedder=_Embedder(),
            )

            denied = service.search(
                query="デザイン",
                access=AccessContext(user_id="u1", guild_id="other"),
            )
            allowed = service.search(
                query="デザイン",
                access=AccessContext(user_id="u1", guild_id="g1"),
            )
            admin = service.search(
                query="111111",
                access=AccessContext(user_id="admin-1"),
            )

            self.assertFalse(denied.authorized)
            self.assertEqual(denied.profiles, tuple())
            self.assertTrue(allowed.authorized)
            self.assertEqual(allowed.profiles[0].display_name, "Alice")
            self.assertEqual(len(allowed.profiles[0].evidence), 1)
            self.assertTrue(admin.authorized)
            self.assertEqual(admin.profiles[0].display_name, "Alice")

    def test_dense_degraded_sparse_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._repo(Path(tmp))
            for profile in self._profiles():
                repo.save_member_profile(profile)
            service = MemberSearchService(
                repository=repo,
                config=self._config(),
                embedder=_FailingEmbedder(),
            )

            result = service.search(
                query="サーバー運用",
                access=AccessContext(user_id="u1", guild_id="g1"),
            )

            self.assertEqual(result.profiles[0].display_name, "Bob")
            self.assertTrue(result.metadata["degraded"])

    def test_workflow_uses_member_search_service_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operations = self._repo(root)
            operations.save_member_profile(self._profiles()[0])
            member_search = MemberSearchService(
                repository=operations,
                config=self._config(),
                embedder=_Embedder(),
            )
            service = WorkflowService(
                repository=FileWorkflowRepository(root_dir=root / "workflow"),
                operations=operations,
                member_search_service=member_search,
            )

            response = service.run(
                WorkRequest(
                    work_type="member_search",
                    instruction="デザイン",
                    access=AccessContext(user_id="u1", guild_id="g1"),
                )
            )

            self.assertEqual(len(response.member_profiles), 1)
            self.assertEqual(response.metadata["route"], "member_search")
            self.assertIn("search_conditions", response.metadata)
            self.assertNotIn("query", response.metadata)

    def test_profile_build_service_generates_and_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            member = DiscordMemberRecord(
                guild_id="g1",
                user_id="111111",
                display_name="Alice",
                roles=("designer",),
                role_ids=("r-design",),
            )
            builder = MemberProfileBuildService(
                repository=repo,
                directory=_Directory([member, replace(member, user_id="bot", is_bot=True)]),
                evidence_source=_EvidenceSource(),
                generator=MemberProfileGenerator(llm=_LLM(), prompts_dir=ROOT / "assets" / "prompts"),
                config=self._config(),
                indexer=MemberProfileIndexService(
                    index_dir=root / "index",
                    embedder=_Embedder(),
                    config=self._config(),
                ),
            )

            run = builder.rebuild_guild(guild_id="g1")
            profiles = repo.list_member_profiles()

            self.assertEqual(run.status, "succeeded")
            self.assertEqual(run.changed, 1)
            self.assertEqual(run.skipped, 1)
            self.assertEqual(profiles[0].skills, ("イベント告知デザイン",))
            self.assertTrue((root / "index" / "member_profiles" / "keyword" / "member_profiles_sparse.json").exists())

    def test_sensitive_text_masking(self) -> None:
        masked = mask_sensitive_text("mail test@example.com token: abcdefghijklmnopqrstuvwxyz")

        self.assertIn("[MASKED_EMAIL]", masked)
        self.assertIn("[MASKED_SECRET]", masked)


if __name__ == "__main__":
    unittest.main()
