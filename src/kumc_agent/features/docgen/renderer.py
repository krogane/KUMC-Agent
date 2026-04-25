from __future__ import annotations

from kumc_agent.domain.models.docgen import DocumentDraft, DocumentPlan, SectionDraft


class MarkdownRenderer:
    def render(self, plan: DocumentPlan) -> str:
        lines = [f"# {plan.title}", ""]
        if plan.purpose:
            lines.extend(["## 目的", plan.purpose, ""])
        if plan.audience:
            lines.extend(["## 対象読者", plan.audience, ""])
        for section in plan.sections:
            lines.extend(self._section(section))
        if plan.fact_checks:
            lines.extend(["## 公開前チェック", *[f"- [{finding.severity}] {finding.message}" for finding in plan.fact_checks], ""])
        citations = []
        for section in plan.sections:
            citations.extend(section.citations)
        if citations:
            lines.extend(["## 根拠", *[f"- {citation.label or citation.chunk_id} {citation.url}".strip() for citation in citations], ""])
        return "\n".join(lines).strip() + "\n"

    def render_draft(self, draft: DocumentDraft) -> str:
        return draft.markdown

    def _section(self, section: SectionDraft) -> list[str]:
        return [f"## {section.title}", section.body.strip(), ""]
