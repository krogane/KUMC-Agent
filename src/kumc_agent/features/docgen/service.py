from __future__ import annotations

from datetime import UTC, datetime

from kumc_agent.domain.models.docgen import (
    DocGenRequest,
    DocumentDraft,
    DocumentPlan,
    SectionDraft,
)
from kumc_agent.features.docgen.fact_check import FactCheckService
from kumc_agent.features.docgen.renderer import MarkdownRenderer
from kumc_agent.utils.hashing import stable_hash


class DocGenService:
    def __init__(
        self,
        *,
        renderer: MarkdownRenderer | None = None,
        fact_checker: FactCheckService | None = None,
    ) -> None:
        self.renderer = renderer or MarkdownRenderer()
        self.fact_checker = fact_checker or FactCheckService()

    def run(self, request: DocGenRequest | None = None) -> DocumentDraft:
        if request is None:
            raise NotImplementedError("Doc generation request is required.")
        sanitized_source, findings = self.fact_checker.inspect(
            request.source_text,
            public=request.public,
        )
        sanitized_instruction, instruction_findings = self.fact_checker.inspect(
            request.instruction,
            public=request.public,
        )
        sanitized_title, title_findings = self.fact_checker.inspect(
            request.title,
            public=request.public,
        )
        findings = tuple((*findings, *instruction_findings, *title_findings))
        safe_request = DocGenRequest(
            title=sanitized_title,
            instruction=sanitized_instruction,
            source_text=sanitized_source,
            doc_type=request.doc_type,
            audience=request.audience,
            purpose=request.purpose,
            citations=request.citations,
            public=request.public,
        )
        sections = self._sections(safe_request, sanitized_source)
        plan = DocumentPlan(
            id=stable_hash(
                f"document-plan:{safe_request.doc_type}:{safe_request.title}:{safe_request.instruction}"
            )[:32],
            title=safe_request.title or "Document draft",
            doc_type=safe_request.doc_type,
            audience=safe_request.audience,
            purpose=safe_request.purpose or safe_request.instruction,
            sections=tuple(sections),
            fact_checks=findings,
            metadata={"public": request.public},
            created_at=datetime.now(UTC),
        )
        markdown = self.renderer.render(plan)
        warnings = tuple(finding.message for finding in findings if finding.severity == "high")
        return DocumentDraft(plan=plan, markdown=markdown, warnings=warnings)

    def _sections(self, request: DocGenRequest, source_text: str) -> list[SectionDraft]:
        if request.doc_type == "weekly_report":
            return [
                SectionDraft("今週の要点", _bullets(source_text, fallback=request.instruction), request.citations),
                SectionDraft("進捗", "関連資料と会話から確認できた進捗を整理してください。", request.citations),
                SectionDraft("次の確認事項", "未確認の日時・担当・公開条件を確認してください。"),
            ]
        if request.doc_type == "decision_memo":
            return [
                SectionDraft("決定したいこと", request.instruction or "決定事項を記載してください。"),
                SectionDraft("根拠", _bullets(source_text, fallback="根拠資料は未取得です。"), request.citations),
                SectionDraft("未決事項", "不足している情報と追加確認先を記載してください。"),
            ]
        if request.doc_type == "announcement":
            return [
                SectionDraft("告知本文", _announcement_body(request, source_text), request.citations),
                SectionDraft("公開前チェックリスト", "- 日時\n- 場所\n- 参加条件\n- 問い合わせ先"),
            ]
        return [
            SectionDraft("概要", request.instruction or "概要を記載してください。"),
            SectionDraft("本文", _bullets(source_text, fallback="関連資料は未取得です。"), request.citations),
            SectionDraft("次の確認事項", "公開前に未確認情報と引用元を確認してください。"),
        ]


def _bullets(text: str, *, fallback: str) -> str:
    lines = [line.strip("-* ・") for line in text.splitlines() if line.strip()]
    if not lines:
        return fallback
    return "\n".join(f"- {line[:240]}" for line in lines[:8])


def _announcement_body(request: DocGenRequest, source_text: str) -> str:
    base = request.instruction.strip() or "告知内容を記載してください。"
    facts = _bullets(source_text, fallback="")
    if facts:
        return "\n".join([base, "", "確認済み情報:", facts])
    return base
