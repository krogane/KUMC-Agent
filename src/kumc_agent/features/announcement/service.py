from __future__ import annotations

from dataclasses import dataclass

from kumc_agent.domain.models.announcement import AnnouncementDraft
from kumc_agent.domain.models.docgen import DocGenRequest
from kumc_agent.features.docgen.service import DocGenService
from kumc_agent.infra.announcement.repository import AnnouncementRepository
from kumc_agent.utils.hashing import stable_hash


@dataclass(frozen=True)
class AnnouncementDraftRequest:
    title: str
    instruction: str
    source_text: str = ""
    medium: str = "discord"
    audience: str = ""
    created_by: str = "agent"


class AnnouncementDraftService:
    def __init__(
        self,
        *,
        repository: AnnouncementRepository,
        docgen: DocGenService,
    ) -> None:
        self.repository = repository
        self.docgen = docgen

    def draft(self, request: AnnouncementDraftRequest) -> AnnouncementDraft:
        document = self.docgen.run(
            DocGenRequest(
                title=request.title,
                instruction=request.instruction,
                source_text=request.source_text,
                doc_type="announcement",
                audience=request.audience,
                purpose="告知下書き",
                public=True,
            )
        )
        status = "needs_review" if document.plan.fact_checks else "draft"
        draft = AnnouncementDraft(
            id=stable_hash(f"announcement:{request.medium}:{document.markdown}")[:32],
            title=document.plan.title,
            body_markdown=document.markdown,
            medium=request.medium,
            audience=request.audience,
            status=status,
            fact_checks=document.plan.fact_checks,
            citations=document.plan.sections[0].citations if document.plan.sections else tuple(),
            created_by=request.created_by,
            metadata={"document_plan_id": document.plan.id},
        )
        return self.repository.save(draft)
