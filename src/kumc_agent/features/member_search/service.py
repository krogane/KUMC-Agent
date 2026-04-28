from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import re
from typing import Any, Protocol, Sequence
import unicodedata

import numpy as np
from langchain_core.documents import Document

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.operations import IndexingRun, MemberProfile
from kumc_agent.domain.models.retrieval import AccessContext, RetrievalQuery
from kumc_agent.domain.ports.embedders import EmbedderPort
from kumc_agent.domain.ports.llms import LLMPort
from kumc_agent.infra.indexing.keyword_inverted_index import (
    build_and_save_keyword_index,
    load_keyword_index,
)
from kumc_agent.infra.operations import OperationsRepository
from kumc_agent.infra.retrieval.faiss import FaissLikeIndex
from kumc_agent.utils.hashing import stable_hash

try:
    from kumc_agent.infra.indexing.sparse_normalizer import SparseNormalizer, SparseNormalizerConfig
except Exception:  # pragma: no cover - depends on optional Sudachi runtime
    @dataclass(frozen=True)
    class SparseNormalizerConfig:  # type: ignore[no-redef]
        sudachi_mode: str = "B"
        use_normalized_form: bool = True
        remove_symbols: bool = True

    class SparseNormalizer:  # type: ignore[no-redef]
        def __init__(self, *, config: SparseNormalizerConfig) -> None:
            self._config = config

        def normalize_tokens(self, text: str) -> list[str]:
            return _simple_tokens(text)

logger = logging.getLogger(__name__)

_USER_MENTION_RE = re.compile(r"<@!?(\d+)>")
_ROLE_MENTION_RE = re.compile(r"<@&(\d+)>")
_DISPLAY_MENTION_RE = re.compile(r"(?<!\w)@([A-Za-z0-9_.\-\u3040-\u30ff\u3400-\u9fff]+)")
_LABELED_DISPLAY_RE = re.compile(r"(?:display|name|表示名|名前)[:：]\s*([^\s,，、]+)", re.IGNORECASE)
_LABELED_ROLE_RE = re.compile(r"(?:role|ロール|役職)[:：]\s*([^\s,，、]+)", re.IGNORECASE)
_EXCLUDE_USER_RE = re.compile(
    r"(?:exclude[_ -]?user|not[_ -]?user|除外ユーザー|除外user)[:：]\s*(<@!?(\d+)>|\d{5,})",
    re.IGNORECASE,
)
_EXCLUDE_ROLE_RE = re.compile(
    r"(?:exclude[_ -]?role|not[_ -]?role|除外ロール|除外role|除外役職)[:：]\s*(<@&(\d+)>|[^\s,，、]+)",
    re.IGNORECASE,
)
_EXCLUDE_TERM_RE = re.compile(r"(?:exclude|without|除外)[:：]\s*([^\s,，、]+)", re.IGNORECASE)
_NEG_ROLE_RE = re.compile(r"(?<!\w)-role[:：]?\s*(<@&(\d+)>|[^\s,，、]+)", re.IGNORECASE)
_BARE_ID_RE = re.compile(r"(?<!\d)(\d{5,})(?!\d)")
_SECRET_RE = re.compile(
    r"(?i)(discord(?:app)?\.com/invite/\S+|[A-Za-z0-9_\-]{24,}\.[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{20,}|api[_-]?key\s*[:=]\s*\S+|token\s*[:=]\s*\S+)"
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\- ]{8,}\d)(?!\d)")
_PRIVATE_IP_RE = re.compile(r"\b(?:10|127)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\b192\.168\.\d{1,3}\.\d{1,3}\b|\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b")
_STUDENT_ID_RE = re.compile(r"(?i)(?:学籍番号|student\s*id)\s*[:：]?\s*[A-Za-z0-9\-]{4,}")
_TOKEN_RE = re.compile(r"[0-9A-Za-z_\-]+|[ぁ-んァ-ン一-龠々ー]+")
_ASSERTIVE_REPLACEMENTS = (
    ("担当できます", "担当候補として確認できます"),
    ("担当可能です", "担当候補として確認できます"),
    ("詳しいです", "関連する根拠があります"),
    ("得意です", "関連する記録があります"),
    ("参加できます", "参加可否の確認が必要です"),
)
_MEMBER_CORPUS_NORMAL = "member_profiles_sparse"
_MEMBER_CORPUS_STEMMING = "member_profiles_stemming"
_PROFILE_VERSION = "member-search-v1"
_MEMBER_INDEX_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class _IndexDocument:
    page_content: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class DiscordMemberRecord:
    guild_id: str
    user_id: str
    display_name: str
    roles: tuple[str, ...] = tuple()
    role_ids: tuple[str, ...] = tuple()
    username: str = ""
    joined_at: datetime | None = None
    is_bot: bool = False
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_fingerprint(self) -> str:
        payload = {
            "guild_id": self.guild_id,
            "user_id": self.user_id,
            "display_name": self.display_name,
            "roles": list(self.roles),
            "role_ids": list(self.role_ids),
            "is_bot": self.is_bot,
            "is_active": self.is_active,
        }
        return stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))


class MemberDirectoryConnector(Protocol):
    def list_members(self, *, guild_id: str) -> list[DiscordMemberRecord]:
        ...


class RagEvidenceSource(Protocol):
    def collect(self, *, member: DiscordMemberRecord, access: AccessContext) -> tuple[dict[str, Any], ...]:
        ...


@dataclass(frozen=True)
class MemberSearchConfig:
    allowed_guild_ids: tuple[str, ...] = tuple()
    admin_user_ids: tuple[str, ...] = tuple()
    exclude_bot_members: bool = True
    exclude_role_names: tuple[str, ...] = tuple()
    max_evidence: int = 6
    search_limit: int = 5
    rrf_k: int = 60
    dense_enabled: bool = True
    sparse_bm25_k1: float = 1.5
    sparse_bm25_b: float = 0.75
    sudachi_mode: str = "B"
    sparse_use_normalized_form: bool = True
    sparse_remove_symbols: bool = True
    profile_prompt_name: str = "member_profile_generation"
    answer_prompt_name: str = "member_search_answer"


@dataclass(frozen=True)
class MemberSearchConditions:
    user_ids: tuple[str, ...] = tuple()
    display_names: tuple[str, ...] = tuple()
    role_ids: tuple[str, ...] = tuple()
    role_names: tuple[str, ...] = tuple()
    exclude_user_ids: tuple[str, ...] = tuple()
    exclude_role_ids: tuple[str, ...] = tuple()
    exclude_role_names: tuple[str, ...] = tuple()
    exclude_terms: tuple[str, ...] = tuple()

    def as_metadata(self) -> dict[str, object]:
        return {
            "user_ids": list(self.user_ids),
            "display_names": list(self.display_names),
            "role_ids": list(self.role_ids),
            "role_names": list(self.role_names),
            "exclude_user_ids": list(self.exclude_user_ids),
            "exclude_role_ids": list(self.exclude_role_ids),
            "exclude_role_names": list(self.exclude_role_names),
            "exclude_terms": list(self.exclude_terms),
        }


@dataclass(frozen=True)
class MemberSearchCandidate:
    profile: MemberProfile
    score: float
    rank: int
    reason: str
    evidence: tuple[dict[str, Any], ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemberSearchResult:
    text: str
    detail_markdown: str
    profiles: tuple[MemberProfile, ...]
    candidates: tuple[MemberSearchCandidate, ...]
    authorized: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class AskServiceEvidenceSource:
    def __init__(self, *, ask_service: Any, max_evidence: int = 6) -> None:
        self._ask_service = ask_service
        self._max_evidence = max(0, max_evidence)

    def collect(self, *, member: DiscordMemberRecord, access: AccessContext) -> tuple[dict[str, Any], ...]:
        if self._ask_service is None or self._max_evidence <= 0:
            return tuple()
        queries = _member_evidence_queries(member)
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for query in queries:
            try:
                response = self._ask_service.ask(
                    RetrievalQuery(text=query, source_filter="all", mode="search_only", access=access)
                )
            except Exception:
                logger.warning("Member evidence RAG query failed: %s", query, exc_info=True)
                continue
            for citation in getattr(response, "citations", ()) or ():
                source_item_id = str(getattr(citation, "source_item_id", "") or "")
                chunk_id = str(getattr(citation, "chunk_id", "") or "")
                key = (source_item_id, chunk_id)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    sanitize_evidence(
                        {
                            "source_type": _citation_source_type(citation),
                            "source_item_id": source_item_id,
                            "chunk_id": chunk_id,
                            "label": str(getattr(citation, "label", "") or ""),
                            "url": str(getattr(citation, "url", "") or ""),
                            "quote": str(getattr(citation, "quote", "") or ""),
                            "access_scope": _citation_access_scope(citation),
                            "score": getattr(citation, "score", None),
                            "metadata": {"query": query},
                        }
                    )
                )
                if len(out) >= self._max_evidence:
                    return tuple(out)
        return tuple(out)


class MemberProfileGenerator:
    def __init__(
        self,
        *,
        llm: LLMPort | None = None,
        prompts_dir: Path | None = None,
        prompt_name: str = "member_profile_generation",
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> None:
        self._llm = llm
        self._prompts_dir = prompts_dir
        self._prompt_name = prompt_name
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    def generate(
        self,
        *,
        member: DiscordMemberRecord,
        evidence: tuple[dict[str, Any], ...],
    ) -> MemberProfile:
        base = _fallback_profile(member=member, evidence=evidence, status="fallback")
        if self._llm is None or not evidence:
            return base
        try:
            system_prompt = _read_prompt(self._prompts_dir, self._prompt_name)
            user_prompt = json.dumps(
                {
                    "discord_member": {
                        "guild_id": member.guild_id,
                        "user_id": member.user_id,
                        "display_name": member.display_name,
                        "roles": list(member.roles),
                        "role_ids": list(member.role_ids),
                    },
                    "evidence": list(evidence),
                },
                ensure_ascii=False,
            )
            raw = self._llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
            )
            payload = _extract_json_object(raw)
            evidence_ids = _evidence_ids(evidence)
            skills, skill_evidence = _clean_evidenced_terms(payload.get("skills"), evidence_ids=evidence_ids)
            interests, interest_evidence = _clean_evidenced_terms(payload.get("interests"), evidence_ids=evidence_ids)
            assignments, assignment_evidence = _clean_evidenced_terms(
                payload.get("past_assignments"),
                evidence_ids=evidence_ids,
            )
            return replace(
                base,
                skills=tuple(skills),
                interests=tuple(interests),
                past_assignments=tuple(assignments),
                metadata=base.metadata
                | {
                    "profile_status": "generated",
                    "generated_by": "llm",
                    "generation_model": payload.get("model") or "",
                    "generation_confidence": payload.get("confidence") or "",
                    "term_evidence": {
                        "skills": skill_evidence,
                        "interests": interest_evidence,
                        "past_assignments": assignment_evidence,
                    },
                },
            )
        except Exception as exc:
            logger.warning("Member profile LLM generation failed: %s", member.user_id, exc_info=True)
            return replace(
                base,
                metadata=base.metadata
                | {
                    "profile_status": "fallback",
                    "generated_by": "fallback",
                    "generation_error": str(exc),
                },
            )


class MemberProfileBuildService:
    def __init__(
        self,
        *,
        repository: OperationsRepository,
        directory: MemberDirectoryConnector,
        evidence_source: RagEvidenceSource | None,
        generator: MemberProfileGenerator,
        config: MemberSearchConfig,
        indexer: "MemberProfileIndexService | None" = None,
    ) -> None:
        self._repository = repository
        self._directory = directory
        self._evidence_source = evidence_source
        self._generator = generator
        self._config = config
        self._indexer = indexer

    def rebuild_guild(self, *, guild_id: str, index_dir: Path | None = None) -> IndexingRun:
        run = IndexingRun(
            id=stable_hash(f"member-profile-build:{guild_id}:{datetime.now(UTC).isoformat()}")[:32],
            source_kind="member_profiles",
            status="running",
            metadata={"guild_id": guild_id},
        )
        try:
            members = self._directory.list_members(guild_id=guild_id)
            existing_profiles = {
                profile.id: profile
                for profile in self._repository.list_member_profiles()
                if str(profile.metadata.get("guild_id") or "") == str(guild_id)
            }
            active_profile_ids: set[str] = set()
            changed = 0
            skipped = 0
            deleted = 0
            for member in members:
                profile_id = stable_hash(f"member_profile:{member.guild_id}:{member.user_id}")[:32]
                if self._should_skip(member):
                    skipped += 1
                    continue
                active_profile_ids.add(profile_id)
                existing = existing_profiles.get(profile_id)
                if (
                    existing is not None
                    and existing.metadata.get("source_fingerprint") == member.source_fingerprint
                    and _is_active_profile(existing)
                ):
                    skipped += 1
                    continue
                access = AccessContext(user_id="", guild_id=member.guild_id, is_admin=True)
                evidence = (
                    self._evidence_source.collect(member=member, access=access)
                    if self._evidence_source is not None
                    else tuple()
                )
                evidence = tuple(sanitize_evidence(item) for item in evidence)
                profile = self._generator.generate(member=member, evidence=evidence)
                self._repository.save_member_profile(profile)
                changed += 1
            for profile_id, profile in existing_profiles.items():
                if profile_id in active_profile_ids or not _is_active_profile(profile):
                    continue
                self._repository.save_member_profile(
                    replace(
                        profile,
                        metadata={
                            **profile.metadata,
                            "profile_status": "inactive",
                            "inactive_reason": "guild_member_missing_or_excluded",
                        },
                    )
                )
                deleted += 1
            index_metadata: dict[str, object] = {}
            if self._indexer is not None:
                index_metadata = self._indexer.rebuild(
                    self._repository.list_member_profiles(),
                    index_dir=index_dir,
                )
            return self._repository.save_indexing_run(
                replace(
                    run,
                    status="succeeded",
                    seen=len(members),
                    changed=changed,
                    skipped=skipped,
                    deleted=deleted,
                    metadata=run.metadata | {"index": index_metadata},
                )
            )
        except Exception as exc:
            return self._repository.save_indexing_run(
                replace(run, status="failed", error=str(exc))
            )

    def _should_skip(self, member: DiscordMemberRecord) -> bool:
        if self._config.exclude_bot_members and member.is_bot:
            return True
        if not member.is_active:
            return True
        excluded = {_normalize_key(role) for role in self._config.exclude_role_names}
        return any(_normalize_key(role) in excluded for role in member.roles)


class MemberProfileIndexService:
    def __init__(
        self,
        *,
        index_dir: Path,
        embedder: EmbedderPort | None,
        config: MemberSearchConfig,
    ) -> None:
        self._index_dir = index_dir / "member_profiles"
        self._embedder = embedder
        self._config = config

    def rebuild(
        self,
        profiles: Sequence[MemberProfile],
        *,
        index_dir: Path | None = None,
    ) -> dict[str, object]:
        target_index_dir = index_dir / "member_profiles" if index_dir is not None else self._index_dir
        index_profiles = [
            _profile_for_index(profile)
            for profile in profiles
            if _is_active_profile(profile)
        ]
        sparse_docs = [
            _IndexDocument(
                page_content=build_profile_text(profile, include_user_id=True),
                metadata={"profile_id": profile.id, "discord_user_id": profile.discord_user_id},
            )
            for profile in index_profiles
        ]
        dense_docs = [
            _IndexDocument(
                page_content=build_profile_text(profile, include_user_id=False),
                metadata={"profile_id": profile.id, "discord_user_id": profile.discord_user_id},
            )
            for profile in index_profiles
        ]
        normal_path = _save_member_sparse_index(
            index_dir=target_index_dir,
            corpus_name=_MEMBER_CORPUS_NORMAL,
            docs=sparse_docs,
            tokenize=lambda text: _simple_tokens(text),
            k1=self._config.sparse_bm25_k1,
            b=self._config.sparse_bm25_b,
        )
        normalizer = _sparse_normalizer(self._config)
        stemming_path = _save_member_sparse_index(
            index_dir=target_index_dir,
            corpus_name=_MEMBER_CORPUS_STEMMING,
            docs=sparse_docs,
            tokenize=lambda text: _safe_stemming_tokens(normalizer, text),
            k1=self._config.sparse_bm25_k1,
            b=self._config.sparse_bm25_b,
        )
        dense_built = False
        if self._embedder is not None and dense_docs:
            texts = [doc.page_content for doc in dense_docs]
            embeddings = self._embedder.embed_documents(texts)
            chunks = [
                Chunk(
                    id=str(doc.metadata["profile_id"]),
                    document_id=str(doc.metadata["profile_id"]),
                    text=doc.page_content,
                    index=index,
                    metadata=dict(doc.metadata) | {"chunk_stage": "member_profile"},
                )
                for index, doc in enumerate(dense_docs)
            ]
            previous_disable_faiss = os.environ.get("KUMC_DISABLE_FAISS_RUNTIME")
            if importlib.util.find_spec("faiss") is None:
                os.environ["KUMC_DISABLE_FAISS_RUNTIME"] = "1"
            try:
                FaissLikeIndex(index_dir=target_index_dir).build(chunks=chunks, embeddings=embeddings)
            finally:
                if previous_disable_faiss is None:
                    os.environ.pop("KUMC_DISABLE_FAISS_RUNTIME", None)
                else:
                    os.environ["KUMC_DISABLE_FAISS_RUNTIME"] = previous_disable_faiss
            dense_built = True
        _write_member_index_metadata(target_index_dir)
        return {
            "profiles": len(index_profiles),
            "schema_version": _MEMBER_INDEX_SCHEMA_VERSION,
            "normal_sparse_index": str(normal_path),
            "stemming_sparse_index": str(stemming_path),
            "dense_built": dense_built,
        }


class MemberSearchService:
    def __init__(
        self,
        *,
        repository: OperationsRepository,
        config: MemberSearchConfig,
        embedder: EmbedderPort | None = None,
        index_dir: Path | None = None,
        llm: LLMPort | None = None,
        prompts_dir: Path | None = None,
    ) -> None:
        self._repository = repository
        self._config = config
        self._embedder = embedder
        self._index_dir = index_dir / "member_profiles" if index_dir is not None else None
        self._llm = llm
        self._prompts_dir = prompts_dir

    def search(self, *, query: str, access: AccessContext, limit: int | None = None) -> MemberSearchResult:
        if not self._is_authorized(access):
            return MemberSearchResult(
                text="権限がありません。",
                detail_markdown="メンバー検索を実行できません。対象情報の有無は表示しません。",
                profiles=tuple(),
                candidates=tuple(),
                authorized=False,
                metadata={"route": "member_search", "authorized": False},
            )
        conditions = extract_conditions(query)
        profiles = _dedupe_profiles(
            [
                _filter_profile_for_response(profile, access, self._config)
                for profile in self._repository.list_member_profiles()
                if self._can_view_profile(profile, access)
            ]
        )
        filtered = self._apply_conditions(profiles, conditions)
        ranked, metadata = self._rank(query=query, profiles=filtered, conditions=conditions)
        selected = ranked[: max(0, limit or self._config.search_limit)]
        candidates = tuple(
            MemberSearchCandidate(
                profile=_filter_profile_for_response(item.profile, access, self._config),
                score=item.score,
                rank=index + 1,
                reason=_candidate_reason(item.profile, conditions, query),
                evidence=_visible_evidence(item.profile.evidence, access, self._config),
                metadata=item.metadata,
            )
            for index, item in enumerate(selected)
        )
        text = self._generate_answer(query=query, candidates=candidates)
        detail = format_candidates_markdown(candidates)
        return MemberSearchResult(
            text=text,
            detail_markdown=detail,
            profiles=tuple(candidate.profile for candidate in candidates),
            candidates=candidates,
            authorized=True,
            metadata={
                "route": "member_search",
                "authorized": True,
                "search_conditions": conditions.as_metadata(),
                **metadata,
            },
        )

    def _is_authorized(self, access: AccessContext) -> bool:
        guild = str(access.guild_id or "")
        if guild and self._config.allowed_guild_ids and guild in self._config.allowed_guild_ids:
            return True
        if not guild and str(access.user_id or "") in self._config.admin_user_ids:
            return True
        return False

    def _can_view_profile(self, profile: MemberProfile, access: AccessContext) -> bool:
        return _is_active_profile(profile) and _can_view_scope(profile.access_scope or {}, access, self._config)

    def _apply_conditions(
        self,
        profiles: list[MemberProfile],
        conditions: MemberSearchConditions,
    ) -> list[MemberProfile]:
        out = profiles
        if conditions.user_ids:
            user_ids = set(conditions.user_ids)
            out = [profile for profile in out if profile.discord_user_id in user_ids]
        if conditions.role_ids or conditions.role_names:
            role_needles = {_normalize_key(value) for value in conditions.role_ids + conditions.role_names}
            out = [
                profile
                for profile in out
                if role_needles & {_normalize_key(role) for role in profile.roles}
                or role_needles & {_normalize_key(role) for role in profile.metadata.get("role_ids", [])}
            ]
        if conditions.display_names:
            needles = [_normalize_key(name) for name in conditions.display_names]
            out = [
                profile
                for profile in out
                if any(needle and needle in _normalize_key(profile.display_name) for needle in needles)
            ]
        if conditions.exclude_user_ids:
            excluded_user_ids = set(conditions.exclude_user_ids)
            out = [profile for profile in out if profile.discord_user_id not in excluded_user_ids]
        if conditions.exclude_role_ids or conditions.exclude_role_names:
            role_needles = {
                _normalize_key(value)
                for value in conditions.exclude_role_ids + conditions.exclude_role_names
            }
            out = [
                profile
                for profile in out
                if not (
                    role_needles & {_normalize_key(role) for role in profile.roles}
                    or role_needles & {_normalize_key(role) for role in profile.metadata.get("role_ids", [])}
                )
            ]
        if conditions.exclude_terms:
            needles = [_normalize_key(term) for term in conditions.exclude_terms]
            out = [
                profile
                for profile in out
                if not any(
                    needle and needle in _normalize_key(build_profile_text(profile, include_user_id=True))
                    for needle in needles
                )
            ]
        return out

    def _rank(
        self,
        *,
        query: str,
        profiles: list[MemberProfile],
        conditions: MemberSearchConditions,
    ) -> tuple[list[MemberSearchCandidate], dict[str, object]]:
        if not profiles:
            return [], {"degraded": False, "candidate_pool": 0, "rank_sources": []}
        normal, normal_degraded, normal_mode = self._sparse_rank(
            query=query,
            profiles=profiles,
            corpus_name=_MEMBER_CORPUS_NORMAL,
            normalize=lambda text: _simple_tokens(text),
        )
        normalizer = _sparse_normalizer(self._config)
        stemming, stemming_degraded, stemming_mode = self._sparse_rank(
            query=query,
            profiles=profiles,
            corpus_name=_MEMBER_CORPUS_STEMMING,
            normalize=lambda text: _safe_stemming_tokens(normalizer, text),
        )
        dense, dense_degraded, dense_mode = self._dense_rank(query=query, profiles=profiles)
        if not (normal or stemming or dense) and not _has_positive_conditions(conditions):
            return [], {
                "degraded": normal_degraded or stemming_degraded or dense_degraded,
                "degraded_reasons": [
                    reason
                    for reason, degraded in (
                        (f"normal_sparse:{normal_mode}", normal_degraded),
                        (f"stemming_sparse:{stemming_mode}", stemming_degraded),
                        (f"dense:{dense_mode}", dense_degraded),
                    )
                    if degraded
                ],
                "candidate_pool": len(profiles),
                "rank_sources": ["normal_sparse", "stemming_sparse"] + ([] if dense_mode == "unavailable" else ["dense"]),
                "rank_source_modes": {
                    "normal_sparse": normal_mode,
                    "stemming_sparse": stemming_mode,
                    "dense": dense_mode,
                },
            }
        fused = _rrf(
            profiles=profiles,
            ranked_sources=(normal, stemming, dense),
            rrf_k=self._config.rrf_k,
        )
        boosted = _boost_exact_matches(fused, conditions)
        candidates = [
            MemberSearchCandidate(
                profile=profile,
                score=score,
                rank=index + 1,
                reason="",
                evidence=tuple(),
                metadata={
                    "score": score,
                    "rank_sources": {
                        "normal_sparse": _rank_of(normal, profile.id),
                        "stemming_sparse": _rank_of(stemming, profile.id),
                        "dense": _rank_of(dense, profile.id),
                    },
                },
            )
            for index, (profile, score) in enumerate(boosted)
        ]
        return candidates, {
            "degraded": normal_degraded or stemming_degraded or dense_degraded,
            "degraded_reasons": [
                reason
                for reason, degraded in (
                    (f"normal_sparse:{normal_mode}", normal_degraded),
                    (f"stemming_sparse:{stemming_mode}", stemming_degraded),
                    (f"dense:{dense_mode}", dense_degraded),
                )
                if degraded
            ],
            "candidate_pool": len(profiles),
            "rank_sources": ["normal_sparse", "stemming_sparse"] + ([] if dense_mode == "unavailable" else ["dense"]),
            "rank_source_modes": {
                "normal_sparse": normal_mode,
                "stemming_sparse": stemming_mode,
                "dense": dense_mode,
            },
        }

    def _sparse_rank(
        self,
        *,
        query: str,
        profiles: list[MemberProfile],
        corpus_name: str,
        normalize: Any,
    ) -> tuple[list[tuple[str, float]], bool, str]:
        if self._index_dir is not None:
            ranked = _keyword_index_rank(
                index_dir=self._index_dir,
                corpus_name=corpus_name,
                query=query,
                profiles=profiles,
                normalize=normalize,
            )
            if ranked is not None:
                return ranked, False, "index"
        return _keyword_rank(query=query, profiles=profiles, normalize=normalize), True, "memory_fallback"

    def _dense_rank(self, *, query: str, profiles: list[MemberProfile]) -> tuple[list[tuple[str, float]], bool, str]:
        if not self._config.dense_enabled or self._embedder is None or not query.strip():
            return [], True, "unavailable"
        try:
            query_vector = self._embedder.embed_query(query)
            if self._index_dir is not None:
                ranked = _dense_index_rank(
                    index_dir=self._index_dir,
                    query_vector=query_vector,
                    profiles=profiles,
                )
                if ranked is not None:
                    return ranked, False, "index"
            texts = [build_profile_text(profile, include_user_id=False) for profile in profiles]
            matrix = self._embedder.embed_documents(texts)
            scores = _cosine_scores(query_vector, matrix)
            ranked = [
                (profiles[index].id, float(score))
                for index, score in enumerate(scores)
                if float(score) > 0.0
            ]
            ranked.sort(key=lambda item: item[1], reverse=True)
            return ranked, True, "memory_fallback"
        except Exception:
            logger.warning("Member dense search failed; sparse fallback is used.")
            return [], True, "unavailable"

    def _generate_answer(self, *, query: str, candidates: tuple[MemberSearchCandidate, ...]) -> str:
        fallback = _template_answer(candidates)
        if self._llm is None or not candidates:
            return fallback
        try:
            system_prompt = _read_prompt(self._prompts_dir, self._config.answer_prompt_name)
            payload = {
                "query": query,
                "candidates": [_candidate_payload(candidate) for candidate in candidates],
            }
            text = self._llm.generate(
                system_prompt=system_prompt,
                user_prompt=json.dumps(payload, ensure_ascii=False),
                temperature=0.0,
                max_output_tokens=1200,
            ).strip()
            return _candidate_safe_answer(text) if text else fallback
        except Exception:
            logger.warning("Member search answer LLM failed; template answer is used.", exc_info=True)
            return fallback


def extract_conditions(query: str) -> MemberSearchConditions:
    text = query or ""
    exclude_user_ids: list[str] = []
    exclude_role_ids: list[str] = []
    exclude_role_names: list[str] = []
    exclude_terms: list[str] = []

    def _exclude_user(match: re.Match[str]) -> str:
        value = match.group(2) or match.group(1)
        if value and value not in exclude_user_ids:
            exclude_user_ids.append(value)
        return " "

    def _exclude_role(match: re.Match[str]) -> str:
        raw = match.group(1)
        role_id = match.group(2)
        if role_id:
            if role_id not in exclude_role_ids:
                exclude_role_ids.append(role_id)
        else:
            cleaned = _clean_terms([raw])
            for value in cleaned:
                if value not in exclude_role_names:
                    exclude_role_names.append(value)
        return " "

    def _exclude_term(match: re.Match[str]) -> str:
        raw = match.group(1)
        if raw.lower().startswith(("user", "role")):
            return match.group(0)
        for value in _clean_terms([raw]):
            if value not in exclude_terms:
                exclude_terms.append(value)
        return " "

    positive_text = _EXCLUDE_USER_RE.sub(_exclude_user, text)
    positive_text = _EXCLUDE_ROLE_RE.sub(_exclude_role, positive_text)
    positive_text = _NEG_ROLE_RE.sub(_exclude_role, positive_text)
    positive_text = _EXCLUDE_TERM_RE.sub(_exclude_term, positive_text)

    user_ids = list(dict.fromkeys(_USER_MENTION_RE.findall(positive_text)))
    role_ids = list(dict.fromkeys(_ROLE_MENTION_RE.findall(positive_text)))
    display_names = list(dict.fromkeys(_DISPLAY_MENTION_RE.findall(positive_text) + _LABELED_DISPLAY_RE.findall(positive_text)))
    role_names = list(dict.fromkeys(_LABELED_ROLE_RE.findall(positive_text)))
    for value in _BARE_ID_RE.findall(positive_text):
        if value not in user_ids and value not in role_ids:
            user_ids.append(value)
    return MemberSearchConditions(
        user_ids=tuple(user_ids),
        display_names=tuple(_clean_terms(display_names)),
        role_ids=tuple(role_ids),
        role_names=tuple(_clean_terms(role_names)),
        exclude_user_ids=tuple(exclude_user_ids),
        exclude_role_ids=tuple(exclude_role_ids),
        exclude_role_names=tuple(exclude_role_names),
        exclude_terms=tuple(exclude_terms),
    )


def build_profile_text(profile: MemberProfile, *, include_user_id: bool = False) -> str:
    evidence_text = " / ".join(
        str(item.get("quote") or item.get("label") or "")
        for item in profile.evidence
        if _safe_evidence_for_index(item)
    )
    parts = [
        f"表示名: {profile.display_name}",
        f"ロール: {', '.join(profile.roles)}",
        f"スキル: {', '.join(profile.skills)}",
        f"興味分野: {', '.join(profile.interests)}",
        f"過去担当: {', '.join(profile.past_assignments)}",
        f"根拠要約: {evidence_text}",
    ]
    if include_user_id:
        parts.append(f"Discord user id: {profile.discord_user_id}")
    return mask_sensitive_text("\n".join(part for part in parts if part.strip()))


def format_candidates_markdown(candidates: Sequence[MemberSearchCandidate]) -> str:
    if not candidates:
        return "MemberProfile はありません。\n\n候補の有無は検索条件と権限に依存します。"
    lines = ["# MemberProfile Candidates"]
    for candidate in candidates:
        profile = candidate.profile
        roles = ", ".join(profile.roles) if profile.roles else "未登録"
        skills = ", ".join(profile.skills) if profile.skills else "未登録"
        assignments = ", ".join(profile.past_assignments) if profile.past_assignments else "未登録"
        lines.append(f"- `{profile.id}` {profile.display_name or profile.discord_user_id or 'unnamed'}")
        lines.append(f"  - roles: {roles}")
        lines.append(f"  - skills: {skills}")
        lines.append(f"  - past_assignments: {assignments}")
        lines.append(f"  - reason: {candidate.reason}")
        for evidence in candidate.evidence[:3]:
            label = evidence.get("label") or evidence.get("source_item_id") or evidence.get("chunk_id") or "根拠"
            quote = evidence.get("quote") or ""
            lines.append(f"  - evidence: {label} {quote}".rstrip())
    lines.append("")
    lines.append("担当決定には本人または運営確認が必要です。")
    return "\n".join(lines)


def sanitize_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "evidence_id",
        "source_type",
        "source_item_id",
        "chunk_id",
        "label",
        "url",
        "quote",
        "access_scope",
        "score",
        "metadata",
    }
    payload = {key: evidence.get(key) for key in allowed if key in evidence}
    payload["evidence_id"] = str(payload.get("evidence_id") or _evidence_id(payload))
    if "quote" in payload:
        payload["quote"] = mask_sensitive_text(str(payload.get("quote") or ""))[:300]
    if "label" in payload:
        payload["label"] = mask_sensitive_text(str(payload.get("label") or ""))[:120]
    scope = payload.get("access_scope")
    if isinstance(scope, dict):
        payload["access_scope"] = dict(scope)
    else:
        payload["access_scope"] = {"admin_only": True, "source_scope_missing": True}
    metadata = dict(payload.get("metadata") or {})
    for key in ("contexts", "context", "raw", "secret", "llm_prompt"):
        metadata.pop(key, None)
    payload["metadata"] = metadata
    return payload


def mask_sensitive_text(text: str) -> str:
    value = _SECRET_RE.sub("[MASKED_SECRET]", text or "")
    value = _EMAIL_RE.sub("[MASKED_EMAIL]", value)
    value = _PHONE_RE.sub("[MASKED_PHONE]", value)
    value = _PRIVATE_IP_RE.sub("[MASKED_IP]", value)
    value = _STUDENT_ID_RE.sub("[MASKED_STUDENT_ID]", value)
    return value


def _candidate_safe_answer(text: str) -> str:
    value = mask_sensitive_text(text)
    for source, replacement in _ASSERTIVE_REPLACEMENTS:
        value = value.replace(source, replacement)
    if "本人または運営確認" not in value and "確認が必要" not in value:
        value = value.rstrip() + "\n担当決定には本人または運営確認が必要です。"
    return value


def _fallback_profile(
    *,
    member: DiscordMemberRecord,
    evidence: tuple[dict[str, Any], ...],
    status: str,
) -> MemberProfile:
    profile_id = stable_hash(f"member_profile:{member.guild_id}:{member.user_id}")[:32]
    return MemberProfile(
        id=profile_id,
        display_name=mask_sensitive_text(member.display_name),
        discord_user_id=member.user_id,
        roles=tuple(mask_sensitive_text(role) for role in member.roles),
        evidence=tuple(sanitize_evidence(item) for item in evidence),
        access_scope={"guild_ids": [member.guild_id]},
        metadata={
            "profile_version": _PROFILE_VERSION,
            "profile_status": status,
            "source_fingerprint": member.source_fingerprint,
            "generated_by": "fallback",
            "guild_id": member.guild_id,
            "role_ids": list(member.role_ids),
            "evidence_count": len(evidence),
            "term_evidence": {
                "skills": {},
                "interests": {},
                "past_assignments": {},
            },
        },
    )


def _member_evidence_queries(member: DiscordMemberRecord) -> list[str]:
    names = [member.display_name, f"<@{member.user_id}>", member.user_id]
    role_text = " ".join(member.roles[:5])
    suffixes = ("担当", "制作", "運営", "開発", "イベント", "得意", "興味")
    queries: list[str] = []
    for name in names:
        if not name:
            continue
        queries.append(name)
        if role_text:
            queries.append(f"{name} {role_text}")
        for suffix in suffixes:
            queries.append(f"{name} {suffix}")
    return list(dict.fromkeys(queries))


def _citation_access_scope(citation: object) -> dict[str, Any]:
    scope = getattr(citation, "access_scope", None)
    if isinstance(scope, dict) and scope:
        return dict(scope)
    metadata = getattr(citation, "metadata", None)
    if isinstance(metadata, dict):
        metadata_scope = metadata.get("access_scope")
        if isinstance(metadata_scope, dict) and metadata_scope:
            return dict(metadata_scope)
    return {"admin_only": True, "source_scope_missing": True}


def _citation_source_type(citation: object) -> str:
    metadata = getattr(citation, "metadata", None)
    if isinstance(metadata, dict):
        value = metadata.get("source_type") or metadata.get("source_kind")
        if value:
            return str(value)
    return "rag"


def _clean_terms(value: object, *, require_evidence: bool = True) -> list[str]:
    if not require_evidence:
        return []
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        text = mask_sensitive_text(str(item or "")).strip()
        text = re.sub(r"\s+", " ", text)
        if not text or len(text) > 80:
            continue
        if text not in out:
            out.append(text)
    return out


def _clean_evidenced_terms(
    value: object,
    *,
    evidence_ids: tuple[str, ...],
) -> tuple[list[str], dict[str, list[str]]]:
    if not evidence_ids or not isinstance(value, (list, tuple)):
        return [], {}
    valid_ids = set(evidence_ids)
    out: list[str] = []
    mapping: dict[str, list[str]] = {}
    for item in value:
        raw_term: object
        raw_ids: object
        if isinstance(item, dict):
            raw_term = item.get("term") or item.get("value") or item.get("name")
            raw_ids = item.get("evidence_ids") or item.get("evidence_id") or item.get("evidence")
        else:
            raw_term = item
            raw_ids = evidence_ids[0] if len(evidence_ids) == 1 else ()
        terms = _clean_terms([raw_term])
        if not terms:
            continue
        ids = _normalize_evidence_refs(raw_ids)
        matched = [evidence_id for evidence_id in ids if evidence_id in valid_ids]
        if not matched:
            continue
        term = terms[0]
        if term not in out:
            out.append(term)
        mapping[term] = matched
    return out, mapping


def _normalize_evidence_refs(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _evidence_ids(evidence: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(str(item.get("evidence_id") or _evidence_id(item)) for item in evidence)


def _evidence_id(evidence: dict[str, Any]) -> str:
    source_item_id = str(evidence.get("source_item_id") or "")
    chunk_id = str(evidence.get("chunk_id") or "")
    if source_item_id or chunk_id:
        return stable_hash(f"member-evidence:{source_item_id}:{chunk_id}")[:32]
    payload = json.dumps(
        {
            "source_type": evidence.get("source_type") or "",
            "label": evidence.get("label") or "",
            "quote": evidence.get("quote") or "",
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return stable_hash(f"member-evidence:{payload}")[:32]


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("member profile generation output must be a JSON object")
    return payload


def _read_prompt(prompts_dir: Path | None, prompt_name: str) -> str:
    if prompts_dir is None:
        return ""
    path = prompts_dir / f"{prompt_name}.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _safe_evidence_for_index(evidence: dict[str, Any]) -> bool:
    quote = str(evidence.get("quote") or evidence.get("label") or "")
    return bool(quote and mask_sensitive_text(quote) == quote)


def _scope_indexable_without_access(scope: dict[str, Any]) -> bool:
    if not scope:
        return False
    if bool(scope.get("admin_only")) or bool(scope.get("source_scope_missing")):
        return False
    visibility = str(scope.get("visibility") or "").strip().lower()
    if visibility:
        return visibility in {"public", "guild"}
    if scope.get("allowed_user_ids") or scope.get("user_ids") or scope.get("role_ids"):
        return False
    return bool(scope.get("guild_ids") or scope.get("guild_id"))


def _profile_for_index(profile: MemberProfile) -> MemberProfile:
    return replace(
        profile,
        evidence=tuple(
            item
            for item in profile.evidence
            if _safe_evidence_for_index(item)
            and _scope_indexable_without_access(dict(item.get("access_scope") or {}))
        ),
    )


def _is_active_profile(profile: MemberProfile) -> bool:
    status = str(profile.metadata.get("profile_status") or "").lower()
    return status not in {"deleted", "inactive", "excluded"}


def _filter_profile_for_response(
    profile: MemberProfile,
    access: AccessContext,
    config: MemberSearchConfig,
) -> MemberProfile:
    return replace(profile, evidence=_visible_evidence(profile.evidence, access, config))


def _visible_evidence(
    evidence: tuple[dict[str, Any], ...],
    access: AccessContext,
    config: MemberSearchConfig,
) -> tuple[dict[str, Any], ...]:
    return tuple(item for item in evidence if _can_view_scope(dict(item.get("access_scope") or {}), access, config))


def _can_view_scope(scope: dict[str, Any], access: AccessContext, config: MemberSearchConfig) -> bool:
    allowed_users = {str(value) for value in scope.get("allowed_user_ids") or []}
    allowed_users |= {str(value) for value in scope.get("user_ids") or []}
    if allowed_users and str(access.user_id or "") not in allowed_users:
        return False
    visibility = str(scope.get("visibility") or "").strip().lower()
    if visibility:
        if visibility == "public":
            return True
        if visibility == "admin":
            return _is_admin_dm(access, config)
        if visibility == "private":
            return bool(allowed_users and str(access.user_id or "") in allowed_users)
        if visibility == "role":
            allowed_roles = {str(value) for value in scope.get("role_ids") or []}
            return bool(allowed_roles and allowed_roles & set(access.role_ids))
        if visibility == "guild":
            scope_guild = str(scope.get("guild_id") or "").strip()
            if not scope_guild:
                guild_ids = {str(value) for value in scope.get("guild_ids") or []}
                return bool(access.guild_id and str(access.guild_id) in guild_ids)
            return bool(access.guild_id and str(access.guild_id) == scope_guild)
        return False
    guild_ids = {str(value) for value in scope.get("guild_ids") or []}
    guild_id = str(scope.get("guild_id") or "").strip()
    if guild_id:
        guild_ids.add(guild_id)
    if guild_ids and str(access.guild_id or "") not in guild_ids:
        if not _is_admin_dm(access, config):
            return False
        if config.allowed_guild_ids and not (guild_ids & set(config.allowed_guild_ids)):
            return False
    if bool(scope.get("admin_only")) and access.guild_id:
        return False
    return True


def _is_admin_dm(access: AccessContext, config: MemberSearchConfig) -> bool:
    return not access.guild_id and str(access.user_id or "") in config.admin_user_ids


def _dedupe_profiles(profiles: list[MemberProfile]) -> list[MemberProfile]:
    latest: dict[tuple[str, str], MemberProfile] = {}
    passthrough: list[MemberProfile] = []
    for profile in profiles:
        guild_id = str(profile.metadata.get("guild_id") or "")
        user_id = str(profile.discord_user_id or "")
        if not user_id:
            passthrough.append(profile)
            continue
        key = (guild_id, user_id)
        current = latest.get(key)
        if current is None or _profile_sort_key(profile) > _profile_sort_key(current):
            latest[key] = profile
    return sorted(
        [*latest.values(), *passthrough],
        key=lambda item: item.display_name or item.id,
    )


def _profile_sort_key(profile: MemberProfile) -> tuple[datetime, str]:
    stamp = profile.updated_at or profile.created_at or datetime.min.replace(tzinfo=UTC)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp, str(profile.metadata.get("source_fingerprint") or "")


def _keyword_index_rank(
    *,
    index_dir: Path,
    corpus_name: str,
    query: str,
    profiles: list[MemberProfile],
    normalize: Any,
) -> list[tuple[str, float]] | None:
    if not _member_index_metadata_valid(index_dir):
        return None
    index = load_keyword_index(index_dir=index_dir, corpus_name=corpus_name)
    if index is None:
        return None
    allowed_ids = {profile.id for profile in profiles}
    query_tokens = normalize(query)
    if not query_tokens:
        return []
    scores = index.get_scores(query_tokens)
    ranked: list[tuple[str, float]] = []
    for doc_index, score in enumerate(scores.tolist()):
        if float(score) <= 0.0 or doc_index >= len(index.docs):
            continue
        metadata = index.docs[doc_index].metadata or {}
        profile_id = str(metadata.get("profile_id") or "")
        if profile_id in allowed_ids:
            ranked.append((profile_id, float(score)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _dense_index_rank(
    *,
    index_dir: Path,
    query_vector: np.ndarray,
    profiles: list[MemberProfile],
) -> list[tuple[str, float]] | None:
    if not _member_index_metadata_valid(index_dir):
        return None
    if not (index_dir / "dense_chunks.jsonl").exists() or not (index_dir / "dense_vectors.npy").exists():
        return None
    allowed_ids = {profile.id for profile in profiles}
    results = FaissLikeIndex(index_dir=index_dir).search(
        query_vector=np.asarray(query_vector, dtype=np.float32),
        top_k=max(1, len(profiles)),
    )
    ranked = [
        (result.chunk.id, float(result.score))
        for result in results
        if result.chunk.id in allowed_ids and float(result.score) > 0.0
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _write_member_index_metadata(index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    path = index_dir / "member_index_metadata.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": _MEMBER_INDEX_SCHEMA_VERSION,
                "profile_version": _PROFILE_VERSION,
                "dense_profile_text": "without_discord_user_id",
                "sparse_profile_text": "with_discord_user_id",
                "evidence_scope_policy": "index_public_or_guild_only",
                "created_at": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def _member_index_metadata_valid(index_dir: Path) -> bool:
    path = index_dir / "member_index_metadata.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    return int(payload.get("schema_version") or 0) == _MEMBER_INDEX_SCHEMA_VERSION


def _keyword_rank(
    *,
    query: str,
    profiles: list[MemberProfile],
    normalize: Any,
) -> list[tuple[str, float]]:
    query_tokens = normalize(query)
    if not query_tokens:
        return []
    query_set = set(query_tokens)
    scored: list[tuple[str, float]] = []
    for profile in profiles:
        tokens = normalize(build_profile_text(profile, include_user_id=True))
        if not tokens:
            continue
        counts = {token: tokens.count(token) for token in set(tokens)}
        overlap = query_set & set(tokens)
        if not overlap:
            continue
        score = sum(counts[token] for token in overlap) / math.sqrt(max(1, len(tokens)))
        scored.append((profile.id, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def _rrf(
    *,
    profiles: list[MemberProfile],
    ranked_sources: tuple[list[tuple[str, float]], ...],
    rrf_k: int,
) -> list[tuple[MemberProfile, float]]:
    by_id = {profile.id: profile for profile in profiles}
    scores: dict[str, float] = {}
    for ranked in ranked_sources:
        for rank, (profile_id, _score) in enumerate(ranked, start=1):
            scores[profile_id] = scores.get(profile_id, 0.0) + (1.0 / (float(rrf_k) + float(rank)))
    if not scores:
        for profile in profiles:
            scores[profile.id] = 0.0
    out = [(by_id[profile_id], score) for profile_id, score in scores.items() if profile_id in by_id]
    out.sort(key=lambda item: (item[1], item[0].updated_at or item[0].created_at or datetime.min), reverse=True)
    return out


def _boost_exact_matches(
    fused: list[tuple[MemberProfile, float]],
    conditions: MemberSearchConditions,
) -> list[tuple[MemberProfile, float]]:
    user_ids = set(conditions.user_ids)
    role_needles = {_normalize_key(value) for value in conditions.role_ids + conditions.role_names}
    display_needles = [_normalize_key(value) for value in conditions.display_names]
    boosted: list[tuple[MemberProfile, float]] = []
    for profile, score in fused:
        bonus = 0.0
        if profile.discord_user_id and profile.discord_user_id in user_ids:
            bonus += 10.0
        if any(needle and needle == _normalize_key(profile.display_name) for needle in display_needles):
            bonus += 5.0
        profile_roles = {_normalize_key(role) for role in profile.roles}
        profile_roles |= {_normalize_key(role) for role in profile.metadata.get("role_ids", [])}
        if role_needles & profile_roles:
            bonus += 2.0
        boosted.append((profile, score + bonus))
    boosted.sort(key=lambda item: item[1], reverse=True)
    return boosted


def _has_positive_conditions(conditions: MemberSearchConditions) -> bool:
    return bool(
        conditions.user_ids
        or conditions.display_names
        or conditions.role_ids
        or conditions.role_names
    )


def _rank_of(ranked: list[tuple[str, float]], profile_id: str) -> int | None:
    for index, (candidate_id, _score) in enumerate(ranked, start=1):
        if candidate_id == profile_id:
            return index
    return None


def _candidate_reason(profile: MemberProfile, conditions: MemberSearchConditions, query: str) -> str:
    reasons: list[str] = []
    if profile.discord_user_id in set(conditions.user_ids):
        reasons.append("Discord user id が一致しています")
    if conditions.display_names and any(
        _normalize_key(name) in _normalize_key(profile.display_name)
        for name in conditions.display_names
    ):
        reasons.append("表示名が一致しています")
    role_needles = {_normalize_key(value) for value in conditions.role_ids + conditions.role_names}
    if role_needles & ({_normalize_key(role) for role in profile.roles} | {_normalize_key(role) for role in profile.metadata.get("role_ids", [])}):
        reasons.append("ロール条件に一致しています")
    query_tokens = set(_simple_tokens(query))
    profile_terms = {
        *_simple_tokens(" ".join(profile.skills)),
        *_simple_tokens(" ".join(profile.interests)),
        *_simple_tokens(" ".join(profile.past_assignments)),
    }
    if query_tokens & profile_terms:
        reasons.append("スキル・興味分野・過去担当に関連する語が一致しています")
    if profile.evidence:
        reasons.append("参照可能な根拠があります")
    return " / ".join(reasons) if reasons else "検索条件との関連が相対的に高い候補です"


def _template_answer(candidates: Sequence[MemberSearchCandidate]) -> str:
    if not candidates:
        return "条件に合うメンバー候補は見つかりませんでした。担当決定には本人または運営確認が必要です。"
    lines = [f"条件に合うメンバー候補は {len(candidates)} 件です。担当決定には本人または運営確認が必要です。"]
    for candidate in candidates:
        profile = candidate.profile
        roles = ", ".join(profile.roles[:5]) if profile.roles else "未登録"
        skills = ", ".join(profile.skills[:5]) if profile.skills else "未登録"
        assignments = ", ".join(profile.past_assignments[:5]) if profile.past_assignments else "未登録"
        evidence_labels = ", ".join(
            str(item.get("label") or item.get("source_item_id") or item.get("chunk_id") or "")
            for item in candidate.evidence[:3]
            if item.get("label") or item.get("source_item_id") or item.get("chunk_id")
        ) or "表示可能な根拠なし"
        lines.append(
            f"- {profile.display_name or profile.discord_user_id}: "
            f"roles={roles} / skills={skills} / past_assignments={assignments} / "
            f"reason={candidate.reason} / evidence={evidence_labels}"
        )
    return "\n".join(lines)


def _candidate_payload(candidate: MemberSearchCandidate) -> dict[str, object]:
    profile = candidate.profile
    return {
        "display_name": profile.display_name,
        "roles": list(profile.roles),
        "skills": list(profile.skills),
        "interests": list(profile.interests),
        "past_assignments": list(profile.past_assignments),
        "reason": candidate.reason,
        "evidence": [
            {
                "label": item.get("label") or "",
                "quote": item.get("quote") or "",
            }
            for item in candidate.evidence
        ],
    }


def _simple_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return [match.group(0) for match in _TOKEN_RE.finditer(normalized)]


def _sparse_normalizer(config: MemberSearchConfig) -> SparseNormalizer:
    return SparseNormalizer(
        config=SparseNormalizerConfig(
            sudachi_mode=config.sudachi_mode,
            use_normalized_form=config.sparse_use_normalized_form,
            remove_symbols=config.sparse_remove_symbols,
        )
    )


def _safe_stemming_tokens(normalizer: SparseNormalizer, text: str) -> list[str]:
    try:
        tokens = normalizer.normalize_tokens(text)
    except Exception:
        tokens = _simple_tokens(text)
    out: list[str] = []
    for token in tokens:
        value = unicodedata.normalize("NFKC", token).casefold()
        if len(value) > 4 and value.endswith("ing"):
            value = value[:-3]
        elif len(value) > 3 and value.endswith("ed"):
            value = value[:-2]
        elif len(value) > 3 and value.endswith("s"):
            value = value[:-1]
        if value:
            out.append(value)
    return out


def _save_member_sparse_index(
    *,
    index_dir: Path,
    corpus_name: str,
    docs: Sequence[_IndexDocument],
    tokenize: Any,
    k1: float,
    b: float,
) -> Path:
    return build_and_save_keyword_index(
        index_dir=index_dir,
        corpus_name=corpus_name,
        docs=[
            Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            for doc in docs
        ],
        tokenize_doc=lambda doc: tokenize(doc.page_content),
        k1=k1,
        b=b,
    )


def _normalize_key(value: object) -> str:
    return "".join(_simple_tokens(str(value or "")))


def _cosine_scores(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = np.asarray(query_vector, dtype=np.float32)
    docs = np.asarray(matrix, dtype=np.float32)
    if query.ndim != 1:
        query = query.reshape(-1)
    if docs.ndim == 1:
        docs = docs.reshape(1, -1)
    if docs.size == 0:
        return np.zeros(0, dtype=np.float32)
    query_norm = np.linalg.norm(query)
    doc_norms = np.linalg.norm(docs, axis=1)
    query_norm = query_norm if query_norm != 0 else 1.0
    doc_norms[doc_norms == 0] = 1.0
    return np.dot(docs, query) / (doc_norms * query_norm)
