from __future__ import annotations

from kumc_agent.domain.models.agentic import ToolSchema


class ToolSchemaRegistry:
    def __init__(self) -> None:
        self._schemas = {
            schema.name: schema
            for schema in (
                ToolSchema(
                    name="circle_rag_search",
                    description="Search KUMC circle information with ACL-aware retrieval.",
                    input_schema={"type": "object", "required": ["query"]},
                    output_schema={"type": "object", "properties": {"citations": {"type": "array"}}},
                    read_only=True,
                ),
                ToolSchema(
                    name="minecraft_wiki_rag_search",
                    description="Search Minecraft Wiki indexed material.",
                    input_schema={"type": "object", "required": ["query"]},
                    output_schema={"type": "object", "properties": {"citations": {"type": "array"}}},
                    read_only=True,
                ),
                ToolSchema(
                    name="member_search",
                    description="Search member profiles and assignment candidates.",
                    input_schema={"type": "object", "required": ["query"]},
                    output_schema={"type": "object", "properties": {"member_profiles": {"type": "array"}}},
                    read_only=True,
                ),
                ToolSchema(
                    name="image_search",
                    description="Search image and asset candidates.",
                    input_schema={"type": "object", "required": ["query"]},
                    output_schema={"type": "object", "properties": {"assets": {"type": "array"}}},
                    read_only=True,
                ),
                ToolSchema(
                    name="task_search",
                    description="Search task records and pending task candidates.",
                    input_schema={"type": "object", "required": ["query"]},
                    output_schema={"type": "object", "properties": {"tasks": {"type": "array"}}},
                    read_only=True,
                ),
                ToolSchema(
                    name="task_candidate_create",
                    description="Create task change candidates only; never mutates approved task masters.",
                    input_schema={"type": "object", "required": ["instruction"]},
                    output_schema={"type": "object", "properties": {"task_candidates": {"type": "array"}}},
                    read_only=False,
                ),
                ToolSchema(
                    name="event_search",
                    description="Search event records and pending event candidates.",
                    input_schema={"type": "object", "required": ["query"]},
                    output_schema={"type": "object", "properties": {"events": {"type": "array"}}},
                    read_only=True,
                ),
                ToolSchema(
                    name="event_candidate_create",
                    description="Create event change candidates only; never mutates approved event masters.",
                    input_schema={"type": "object", "required": ["instruction"]},
                    output_schema={"type": "object", "properties": {"event_candidates": {"type": "array"}}},
                    read_only=False,
                ),
                ToolSchema(
                    name="server_operation_candidate_create",
                    description="Create Minecraft server operation dry-run candidates only.",
                    input_schema={"type": "object", "required": ["instruction"]},
                    output_schema={"type": "object", "properties": {"server_operations": {"type": "array"}}},
                    read_only=False,
                ),
                ToolSchema(
                    name="approval_candidate_create",
                    description="Create approval records or approval targets for existing candidates.",
                    input_schema={"type": "object", "required": ["target_type", "target_id"]},
                    output_schema={"type": "object", "properties": {"approvals": {"type": "array"}}},
                    read_only=False,
                ),
            )
        }

    def list(self) -> tuple[ToolSchema, ...]:
        return tuple(self._schemas.values())

    def get(self, name: str) -> ToolSchema:
        return self._schemas[name]
