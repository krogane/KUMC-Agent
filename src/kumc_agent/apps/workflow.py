from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kumc_agent.apps.foundation import build_foundation_app_context
from kumc_agent.apps.retrieval import build_retrieval_app_context
from kumc_agent.features.announcement import AnnouncementDraftService
from kumc_agent.features.docgen.service import DocGenService
from kumc_agent.features.image_search import (
    ImageSearchConfig,
    ImageSearchService,
)
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
from kumc_agent.features.task_management import (
    DiscordTaskNotificationSender,
    TaskAccessPolicy,
    TaskExtractionService,
)
from kumc_agent.features.event_management import (
    DiscordEventNotificationSender,
    EventAccessPolicy,
    EventExtractionService,
)
from kumc_agent.features.minecraft import MinecraftSupportService, settings_from_runtime
from kumc_agent.features.minecraft.access import ServerManagementAccessPolicy
from kumc_agent.features.workflow import WorkflowService
from kumc_agent.infra.connectors.discord_members import DiscordMemberDirectoryConnector
from kumc_agent.infra.embeddings.gemini import GeminiEmbedder
from kumc_agent.infra.embeddings.local import LocalEmbedder
from kumc_agent.infra.llm.gemini import GeminiLLM
from kumc_agent.infra.announcement import build_announcement_repository
from kumc_agent.infra.minecraft import build_server_operation_repository
from kumc_agent.infra.minecraft.executor import ServerOperationExecutorRegistry
from kumc_agent.infra.operations import build_operations_repository
from kumc_agent.infra.workflow import build_workflow_repository


@dataclass(frozen=True)
class WorkflowAppContext:
    workflow: WorkflowService
    member_profile_builder: MemberProfileBuildService | None = None


def build_workflow_app_context(*, base_dir: Path | None = None) -> WorkflowAppContext:
    foundation = build_foundation_app_context(base_dir=base_dir)
    retrieval = build_retrieval_app_context(base_dir=base_dir)
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
    member_profile_guild_ids = tuple(
        str(value)
        for value in foundation.config.security.effective_member_profile_guild_ids()
    )
    member_search_allowed_guild_ids = tuple(
        dict.fromkeys(
            str(value)
            for value in (
                foundation.config.security.discord_guild_allow_list
                + foundation.config.security.effective_member_profile_guild_ids()
            )
        )
    )
    member_search_config = MemberSearchConfig(
        allowed_guild_ids=member_search_allowed_guild_ids,
        admin_user_ids=tuple(str(value) for value in foundation.config.security.maintenance_command_author_ids),
        search_limit=foundation.config.features.retrieval.top_k,
        rrf_k=foundation.config.features.retrieval.rrf_k,
        sparse_bm25_k1=foundation.config.features.retrieval.sparse_bm25_k1,
        sparse_bm25_b=foundation.config.features.retrieval.sparse_bm25_b,
        sudachi_mode=foundation.config.features.retrieval.sudachi_mode,
        sparse_use_normalized_form=foundation.config.features.retrieval.sparse_use_normalized_form,
        sparse_remove_symbols=foundation.config.features.retrieval.sparse_remove_symbols,
        exclude_role_names=tuple(foundation.config.features.member_search.exclude_role_names),
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
    image_search = ImageSearchService(
        repository=operations_repository,
        config=ImageSearchConfig(
            limit=foundation.config.features.image_search.limit,
            dense_top_k=foundation.config.features.image_search.dense_top_k,
            feature_top_k=foundation.config.features.image_search.feature_top_k,
            rrf_k=foundation.config.features.image_search.rrf_k,
            ocr_text_char_limit=foundation.config.features.image_search.ocr_text_char_limit,
            surrounding_text_char_limit=(
                foundation.config.features.image_search.surrounding_text_char_limit
            ),
            ocr_model=(
                foundation.config.features.image_search.ocr_model
                or foundation.config.integrations.drive.pdf_ocr_model_path
            ),
            caption_model=(
                foundation.config.features.image_search.caption_model
                or foundation.config.providers.llm.gemini_model
            ),
            feature_model=foundation.config.features.image_search.feature_model,
            feature_dimensions=foundation.config.features.image_search.feature_dimensions,
            duplicate_group_limit=foundation.config.features.image_search.duplicate_group_limit,
        ),
        embedder=embedder,
        index_dir=foundation.config.app.index_dir / "image_search",
        allowed_guild_ids=tuple(str(value) for value in foundation.config.security.discord_guild_allow_list),
        admin_user_ids=tuple(str(value) for value in foundation.config.security.maintenance_command_author_ids),
    ) if foundation.config.features.image_search.enabled else None
    member_profile_builder = MemberProfileBuildService(
        repository=operations_repository,
        directory=DiscordMemberDirectoryConnector(
            bot_token=foundation.config.integrations.discord.bot_token,
            allowed_guild_ids=member_profile_guild_ids,
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
    task_admin_user_ids = tuple(
        dict.fromkeys(
            [
                *[
                    str(value)
                    for value in foundation.config.security.maintenance_command_author_ids
                ],
                *foundation.config.task_management.admin_user_ids,
            ]
        )
    )
    task_notification_sender = (
        DiscordTaskNotificationSender(
            bot_token=foundation.config.integrations.discord.bot_token,
        )
        if foundation.config.integrations.discord.bot_token
        and foundation.config.task_management.notification_channel_id
        else None
    )
    event_admin_user_ids = tuple(
        dict.fromkeys(
            [
                *[
                    str(value)
                    for value in foundation.config.security.maintenance_command_author_ids
                ],
                *foundation.config.event_management.admin_user_ids,
            ]
        )
    )
    event_notification_sender = (
        DiscordEventNotificationSender(
            bot_token=foundation.config.integrations.discord.bot_token,
        )
        if foundation.config.integrations.discord.bot_token
        and foundation.config.event_management.notification_channel_id
        else None
    )
    return WorkflowAppContext(
        workflow=WorkflowService(
            repository=repository,
            ask_service=retrieval.ask,
            audit_log=foundation.audit_log,
            docgen=docgen,
            announcement=AnnouncementDraftService(
                repository=announcement_repository,
                docgen=docgen,
            ),
            minecraft=MinecraftSupportService(
                repository=server_operation_repository,
                feature_flags=foundation.feature_flags,
                access_policy=ServerManagementAccessPolicy(
                    admin_user_ids=tuple(
                        str(value)
                        for value in foundation.config.security.maintenance_command_author_ids
                    )
                ),
                settings=settings_from_runtime(foundation.config.server_management),
                executor=ServerOperationExecutorRegistry(
                    config=settings_from_runtime(foundation.config.server_management)
                ),
            ),
            operations=operations_repository,
            member_search_service=member_search,
            image_search_service=image_search,
            image_search_enabled=foundation.config.features.image_search.enabled,
            task_extractor=TaskExtractionService(
                llm=llm,
                prompts_dir=prompts_dir,
                prompt_name=foundation.config.task_management.prompt_name,
                model_name=foundation.config.providers.llm.gemini_model,
            ),
            task_access_policy=TaskAccessPolicy(
                admin_user_ids=task_admin_user_ids,
                admin_role_ids=tuple(foundation.config.task_management.admin_role_ids),
            ),
            task_notification_sender=task_notification_sender,
            task_notification_channel_id=foundation.config.task_management.notification_channel_id,
            task_approval_batch_interval_days=(
                foundation.config.task_management.approval_batch_interval_days
            ),
            task_due_soon_notice_days=foundation.config.task_management.due_soon_notice_days,
            event_extractor=EventExtractionService(
                llm=llm,
                prompts_dir=prompts_dir,
                prompt_name=foundation.config.event_management.prompt_name,
                model_name=foundation.config.providers.llm.gemini_model,
            ),
            event_access_policy=EventAccessPolicy(
                admin_user_ids=event_admin_user_ids,
                admin_role_ids=tuple(foundation.config.event_management.admin_role_ids),
            ),
            event_notification_sender=event_notification_sender,
            event_notification_channel_id=foundation.config.event_management.notification_channel_id,
            event_approval_batch_interval_days=(
                foundation.config.event_management.approval_batch_interval_days
            ),
            event_notification_before_days=foundation.config.event_management.notification_before_days,
            event_timezone=foundation.config.event_management.timezone,
            llm=llm,
            prompts_dir=prompts_dir,
            llm_model_name=foundation.config.providers.llm.gemini_model,
        ),
        member_profile_builder=member_profile_builder,
    )
