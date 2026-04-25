from __future__ import annotations

from kumc_agent.domain.models.agentic import ToolSchema


class ToolSchemaRegistry:
    def __init__(self) -> None:
        self._schemas = {
            schema.name: schema
            for schema in (
                ToolSchema(
                    name="search_documents",
                    description="Search indexed documents with ACL-aware retrieval.",
                    input_schema={"query": "string", "source_filter": "string"},
                    output_schema={"citations": "array", "summary": "string"},
                    read_only=True,
                ),
                ToolSchema(
                    name="read_chunks",
                    description="Read retrieved chunks through citation detail.",
                    input_schema={"chunk_ids": "array"},
                    output_schema={"notes": "array"},
                    read_only=True,
                ),
                ToolSchema(
                    name="compare_evidence",
                    description="Compare retrieved evidence and identify missing facts.",
                    input_schema={"evidence_items": "array", "success_criteria": "array"},
                    output_schema={"verified": "array", "missing": "array"},
                    read_only=True,
                ),
                ToolSchema(
                    name="search_tasks",
                    description="Search task master records.",
                    input_schema={"query": "string", "status": "string"},
                    output_schema={"tasks": "array"},
                    read_only=True,
                ),
                ToolSchema(
                    name="search_events",
                    description="Search event master records.",
                    input_schema={"query": "string", "status": "string"},
                    output_schema={"events": "array"},
                    read_only=True,
                ),
            )
        }

    def list(self) -> tuple[ToolSchema, ...]:
        return tuple(self._schemas.values())

    def get(self, name: str) -> ToolSchema:
        return self._schemas[name]
