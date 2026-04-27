from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kumc_agent.apps.agentic import build_agentic_app_context
from kumc_agent.apps.foundation import build_foundation_app_context
from kumc_agent.apps.retrieval import build_retrieval_app_context
from kumc_agent.features.announcement import AnnouncementDraftService
from kumc_agent.features.docgen.service import DocGenService
from kumc_agent.features.member_search import (
    MemberProfileBuildService,
    MemberSearchConfig,
    MemberSearchService,
)
from kumc_agent.features.member_search.service import (
    AskServiceEvidenceSource,
    MemberProfileGenerator,
    MemberProfileIndexService,
)
from kumc_agent.features.minecraft import MinecraftSupportService
from kumc_agent.features.workflow import WorkflowService
from kumc_agent.infra.connectors.discord_members import DiscordMemberDirectoryConnector
from kumc_agent.infra.embeddings.gemini import GeminiEmbedder
from kumc_agent.infra.embeddings.local import LocalEmbedder
from kumc_agent.infra.llm.gemini import GeminiLLM
from kumc_agent.infra.announcement import build_announcement_repository
from kumc_agent.infra.minecraft import build_server_operation_repository
from kumc_agent.infra.operations import build_operations_repository
from kumc_agent.infra.workflow import build_workflow_repository


@dataclass(frozen=True)
class WorkflowAppContext:
    workflow: WorkflowService
    member_profile_builder: MemberProfileBuildService | None = None


def build_workflow_app_context(*, base_dir: Path | None = None) -> WorkflowAppContext:
    foundation = build_foundation_app_context(base_dir=base_dir)
    retrieval = build_retrieval_app_context(base_dir=base_dir)
    agentic = build_agentic_app_context(base_dir=base_dir)
    repository = build_workflow_repository(
        postgres=foundation.postgres,
        fallback_dir=foundation.config.base_dir / "data" / "workflow",
    )
    announcement_repository = build_announcement_repository(
        postgres=foundation.postgres,
        fallback_dir=foundation.config.base_dir / "data" / "announcement",
    )
    server_operation_repository = build_server_operation_repository(
        postgres=foundation.postgres,
        fallback_dir=foundation.config.base_dir / "data" / "minecraft",
    )
    operations_repository = build_operations_repository(
        postgres=foundation.postgres,
        fallback_dir=foundation.config.base_dir / "data" / "operations",
    )
    member_search_config = MemberSearchConfig(
        allowed_guild_ids=tuple(str(value) for value in foundation.config.security.discord_guild_allow_list),
        admin_user_ids=tuple(str(value) for value in foundation.config.security.maintenance_command_author_ids),
        search_limit=foundation.config.features.retrieval.top_k,
        rrf_k=foundation.config.features.retrieval.rrf_k,
        sparse_bm25_k1=foundation.config.features.retrieval.sparse_bm25_k1,
        sparse_bm25_b=foundation.config.features.retrieval.sparse_bm25_b,
        sudachi_mode=foundation.config.features.retrieval.sudachi_mode,
        sparse_use_normalized_form=foundation.config.features.retrieval.sparse_use_normalized_form,
        sparse_remove_symbols=foundation.config.features.retrieval.sparse_remove_symbols,
    )
    dimensions = foundation.config.providers.embeddings.dimensions
    if (
        foundation.config.providers.embeddings.provider == "gemini"
        and foundation.config.integrations.gemini_api_key
    ):
        embedder = GeminiEmbedder(
            api_key=foundation.config.integrations.gemini_api_key,
            model_name=foundation.config.providers.embeddings.model,
            dimensions=dimensions,
            requests_per_minute=foundation.config.integrations.gemini_embedding_requests_per_minute,
        )
    else:
        local_model = (
            foundation.config.providers.embeddings.model
            if foundation.config.providers.embeddings.provider != "gemini"
            else ""
        )
        embedder = LocalEmbedder(model_name=local_model, dimensions=dimensions)
    llm = (
        GeminiLLM(
            api_key=foundation.config.integrations.gemini_api_key,
            model=foundation.config.providers.llm.gemini_model,
            requests_per_minute=foundation.config.integrations.gemini_requests_per_minute,
        )
        if foundation.config.providers.llm.provider == "gemini"
        and foundation.config.integrations.gemini_api_key
        else None
    )
    prompts_dir = foundation.config.base_dir / "assets" / "prompts"
    member_search = MemberSearchService(
        repository=operations_repository,
        config=member_search_config,
        embedder=embedder,
        index_dir=foundation.config.app.index_dir,
        llm=llm,
        prompts_dir=prompts_dir,
    )
    member_profile_builder = MemberProfileBuildService(
        repository=operations_repository,
        directory=DiscordMemberDirectoryConnector(
            bot_token=foundation.config.integrations.discord.bot_token,
            allowed_guild_ids=tuple(str(value) for value in foundation.config.security.discord_guild_allow_list),
        ),
        evidence_source=AskServiceEvidenceSource(
            ask_service=retrieval.ask,
            max_evidence=member_search_config.max_evidence,
        ),
        generator=MemberProfileGenerator(
            llm=llm,
            prompts_dir=prompts_dir,
            prompt_name=member_search_config.profile_prompt_name,
            temperature=0.0,
        ),
        config=member_search_config,
        indexer=MemberProfileIndexService(
            index_dir=foundation.config.app.index_dir,
            embedder=embedder,
            config=member_search_config,
        ),
    )
    docgen = DocGenService()
    return WorkflowAppContext(
        workflow=WorkflowService(
            repository=repository,
            ask_service=retrieval.ask,
            audit_log=foundation.audit_log,
            agentic_search=agentic.agentic_search,
            docgen=docgen,
            announcement=AnnouncementDraftService(
                repository=announcement_repository,
                docgen=docgen,
            ),
            minecraft=MinecraftSupportService(
                repository=server_operation_repository,
                feature_flags=foundation.feature_flags,
            ),
            operations=operations_repository,
            member_search_service=member_search,
        ),
        member_profile_builder=member_profile_builder,
    )
