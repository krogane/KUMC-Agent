from __future__ import annotations

import asyncio
import io
import json
import logging

from kumc_agent.domain.models.health import HealthReport
from kumc_agent.domain.models.integrated_input import IntegratedInputRequest, IntegratedInputResponse
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.domain.models.source import BackfillScope
from kumc_agent.domain.models.workflow import WorkRequest

logger = logging.getLogger(__name__)


def _format_health_report(report: HealthReport) -> str:
    payload = report.as_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(text) <= 1800:
        return f"```json\n{text}\n```"
    return f"status={report.status} components={len(report.components)}"


def _format_markdown_attachment(*, content: str, filename: str):
    import discord

    payload = content.encode("utf-8")
    return discord.File(
        fp=io.BytesIO(payload),
        filename=filename,
    )


def _format_detail_attachment(*, content: str):
    return _format_markdown_attachment(
        content=content,
        filename="kumc-agent-answer-detail.md",
    )


def create_bot(
    *,
    foundation_context: object,
    retrieval_context: object,
    agentic_context: object,
    workflow_context: object,
    automation_context: object,
    ingestion_context: object,
    integrated_input_context: object | None = None,
):
    import discord
    from discord import app_commands
    from discord.ext import commands

    globals()["discord"] = discord
    globals()["app_commands"] = app_commands

    intents = discord.Intents.default()
    bot = commands.Bot(command_prefix="!", intents=intents)
    admin = app_commands.Group(name="admin", description="KUMC-Agent admin actions")

    def _is_authorized(interaction: discord.Interaction) -> bool:
        allowed_users = set(foundation_context.config.security.maintenance_command_author_ids)
        if allowed_users and int(interaction.user.id) not in allowed_users:
            return False
        allowed_guilds = set(foundation_context.config.security.discord_guild_allow_list)
        guild_id = interaction.guild_id
        if allowed_guilds and (guild_id is None or int(guild_id) not in allowed_guilds):
            return False
        return True

    def _access_context(interaction: discord.Interaction) -> AccessContext:
        roles = tuple(
            str(getattr(role, "id", ""))
            for role in getattr(interaction.user, "roles", [])
            if getattr(role, "id", None)
        )
        return AccessContext(
            user_id=str(interaction.user.id),
            guild_id=str(interaction.guild_id or ""),
            role_ids=roles,
            is_admin=_is_authorized(interaction),
        )

    def _task_approval_view(target_id: str):
        if not target_id:
            return None

        class TaskApprovalView(discord.ui.View):
            def __init__(self, candidate_id: str) -> None:
                super().__init__(timeout=86400)
                self.candidate_id = candidate_id

            async def _run(self, interaction: discord.Interaction, action: str) -> None:
                await interaction.response.defer(ephemeral=True, thinking=True)
                response = await asyncio.to_thread(
                    workflow_context.workflow.approval,
                    action=action,
                    target_type="task",
                    target_id=self.candidate_id,
                    comment=f"discord_component:{action}",
                    access=_access_context(interaction),
                )
                kwargs = {"content": response.text, "ephemeral": True}
                if response.detail_markdown and len(response.detail_markdown) > len(response.text):
                    kwargs["file"] = _format_markdown_attachment(
                        content=response.detail_markdown,
                        filename="kumc-agent-task-approval-detail.md",
                    )
                await interaction.followup.send(**kwargs)

            @discord.ui.button(label="Approve", style=discord.ButtonStyle.success, custom_id="task:approve")
            async def approve_button(self, interaction: discord.Interaction, button) -> None:
                await self._run(interaction, "approve")

            @discord.ui.button(label="Reject", style=discord.ButtonStyle.danger, custom_id="task:reject")
            async def reject_button(self, interaction: discord.Interaction, button) -> None:
                await self._run(interaction, "reject")

            @discord.ui.button(label="Show", style=discord.ButtonStyle.secondary, custom_id="task:show")
            async def show_button(self, interaction: discord.Interaction, button) -> None:
                await self._run(interaction, "show")

        return TaskApprovalView(target_id)

    def _first_candidate_id(response: IntegratedInputResponse) -> str:
        candidates = (
            tuple(response.task_candidates or tuple())
            + tuple(response.task_change_candidates or tuple())
        )
        if not candidates:
            return ""
        first = candidates[0]
        if isinstance(first, dict):
            return str(first.get("id") or first.get("operation_id") or "")
        return str(getattr(first, "id", "") or getattr(first, "operation_id", ""))

    async def _send_integrated_response(
        interaction: discord.Interaction,
        response: IntegratedInputResponse,
    ) -> None:
        kwargs = {"content": response.text or "結果がありません。", "ephemeral": True}
        if response.detail_markdown and (
            len(response.detail_markdown) > len(response.text or "")
            or len(response.text or "") > 1800
        ):
            kwargs["file"] = _format_detail_attachment(content=response.detail_markdown)
            if len(str(kwargs["content"])) > 1800:
                kwargs["content"] = response.text[:1700].rstrip() + "..."
        candidate_id = _first_candidate_id(response)
        if candidate_id:
            kwargs["view"] = _task_approval_view(candidate_id)
        await interaction.followup.send(**kwargs)

    def _event_approval_view(target_id: str, *, batch_id: str = ""):
        if not target_id:
            return None

        class EventApprovalView(discord.ui.View):
            def __init__(self, candidate_id: str) -> None:
                super().__init__(timeout=86400)
                self.candidate_id = candidate_id
                nonce = "v1"
                for label, action, style in (
                    ("Approve", "approve", discord.ButtonStyle.success),
                    ("Reject", "reject", discord.ButtonStyle.danger),
                    ("Edit", "edit", discord.ButtonStyle.primary),
                    ("Evidence", "show", discord.ButtonStyle.secondary),
                    ("Diff", "show", discord.ButtonStyle.secondary),
                ):
                    button = discord.ui.Button(
                        label=label,
                        style=style,
                        custom_id=f"event:{candidate_id}:{action}:{batch_id or 'single'}:{nonce}",
                    )
                    if action == "edit":
                        button.callback = self._edit_callback
                    else:
                        button.callback = self._button_callback(action)
                    self.add_item(button)

            async def _run(self, interaction: discord.Interaction, action: str) -> None:
                await interaction.response.defer(ephemeral=True, thinking=True)
                response = await asyncio.to_thread(
                    workflow_context.workflow.approval,
                    action=action,
                    target_type="event",
                    target_id=self.candidate_id,
                    comment=f"discord_component:{action}:batch={batch_id}",
                    access=_access_context(interaction),
                )
                kwargs = {"content": response.text, "ephemeral": True}
                if response.detail_markdown and len(response.detail_markdown) > len(response.text):
                    kwargs["file"] = _format_markdown_attachment(
                        content=response.detail_markdown,
                        filename="kumc-agent-event-approval-detail.md",
                    )
                await interaction.followup.send(**kwargs)

            def _button_callback(self, action: str):
                async def _callback(interaction: discord.Interaction) -> None:
                    await self._run(interaction, action)

                return _callback

            async def _edit_callback(self, interaction: discord.Interaction) -> None:
                view = self

                class EventEditModal(discord.ui.Modal, title="Edit Event Candidate"):
                    comment = discord.ui.TextInput(
                        label="修正内容",
                        style=discord.TextStyle.paragraph,
                        required=True,
                        max_length=1000,
                    )

                    async def on_submit(self, modal_interaction: discord.Interaction) -> None:
                        await modal_interaction.response.defer(ephemeral=True, thinking=True)
                        response = await asyncio.to_thread(
                            workflow_context.workflow.approval,
                            action="edit",
                            target_type="event",
                            target_id=view.candidate_id,
                            comment=str(self.comment.value),
                            access=_access_context(modal_interaction),
                        )
                        kwargs = {"content": response.text, "ephemeral": True}
                        if response.detail_markdown and len(response.detail_markdown) > len(response.text):
                            kwargs["file"] = _format_markdown_attachment(
                                content=response.detail_markdown,
                                filename="kumc-agent-event-edit-detail.md",
                            )
                        await modal_interaction.followup.send(**kwargs)

                await interaction.response.send_modal(EventEditModal())

        return EventApprovalView(target_id)

    @admin.command(name="action", description="Run an approved admin action")
    @app_commands.describe(action="Admin action to run")
    @app_commands.choices(
        action=[
            app_commands.Choice(name="health", value="health"),
            app_commands.Choice(name="readiness", value="readiness"),
            app_commands.Choice(name="sync", value="sync"),
            app_commands.Choice(name="eval", value="eval"),
            app_commands.Choice(name="feature_flags", value="feature_flags"),
            app_commands.Choice(name="permissions", value="permissions"),
            app_commands.Choice(name="reindex", value="reindex"),
            app_commands.Choice(name="cost_report", value="cost_report"),
            app_commands.Choice(name="member_profiles", value="member_profiles"),
        ]
    )
    async def admin_action(
        interaction: discord.Interaction,
        action: app_commands.Choice[str],
        scope: str = "",
    ) -> None:
        if not _is_authorized(interaction):
            await interaction.response.send_message("権限がありません。", ephemeral=True)
            return
        if action.value not in {
            "health",
            "readiness",
            "sync",
            "eval",
            "feature_flags",
            "permissions",
            "reindex",
            "cost_report",
            "member_profiles",
        }:
            await interaction.response.send_message("未対応の action です。", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True, thinking=True)
        if action.value == "health":
            report = await asyncio.to_thread(
                foundation_context.health.check,
                actor_id=str(interaction.user.id),
                actor_type="discord_user",
            )
            await interaction.followup.send(_format_health_report(report), ephemeral=True)
            return
        if action.value == "readiness":
            readiness = await asyncio.to_thread(automation_context.readiness.report)
            payload = readiness.as_dict()
            summary = readiness.summary
            filename = "kumc-agent-production-readiness.json"
        elif action.value in {"sync", "reindex"}:
            if scope.strip() == "member_profiles":
                guild_ids = [str(value) for value in foundation_context.config.security.discord_guild_allow_list]
                results = await asyncio.to_thread(
                    lambda: [
                        workflow_context.member_profile_builder.rebuild_guild(guild_id=guild_id).__dict__
                        for guild_id in guild_ids
                        if workflow_context.member_profile_builder is not None
                    ]
                )
                payload = {
                    "action": action.value,
                    "source_kind": "member_profiles",
                    "results": results,
                    "metadata": {"guild_ids": guild_ids},
                }
                summary = f"{action.value} member_profiles completed: {len(results)} guild(s)"
                filename = "kumc-agent-member-profiles.json"
            else:
                source_kinds = (scope.strip(),) if scope.strip() else tuple()
                results = await asyncio.to_thread(
                    lambda: asyncio.run(
                        ingestion_context.service.backfill_many(
                            source_kinds=source_kinds,
                            scope=BackfillScope(force=action.value == "reindex"),
                        )
                    )
                )
                payload = {
                    "action": action.value,
                    "results": [result.__dict__ for result in results],
                }
                summary = f"{action.value} completed: {len(results)} source(s)"
                filename = "kumc-agent-admin-sync.json"
        elif action.value == "member_profiles":
            guild_ids = [scope.strip()] if scope.strip() else [
                str(value) for value in foundation_context.config.security.discord_guild_allow_list
            ]
            results = await asyncio.to_thread(
                lambda: [
                    workflow_context.member_profile_builder.rebuild_guild(guild_id=guild_id).__dict__
                    for guild_id in guild_ids
                    if workflow_context.member_profile_builder is not None
                ]
            )
            payload = {
                "action": "member_profiles",
                "results": results,
                "metadata": {"guild_ids": guild_ids},
            }
            summary = f"member_profiles completed: {len(results)} guild(s)"
            filename = "kumc-agent-member-profiles.json"
        elif action.value == "eval":
            readiness = await asyncio.to_thread(automation_context.readiness.report)
            payload = {
                "action": "eval",
                "mode": "local_harness",
                "readiness": readiness.as_dict(),
            }
            summary = readiness.summary
            filename = "kumc-agent-admin-eval.json"
        elif action.value == "feature_flags":
            payload = foundation_context.feature_flags.modes()
            summary = "feature flags"
            filename = "kumc-agent-feature-flags.json"
        elif action.value == "permissions":
            payload = {
                "maintenance_command_author_ids": foundation_context.config.security.maintenance_command_author_ids,
                "discord_guild_allow_list": foundation_context.config.security.discord_guild_allow_list,
                "admin_configured": bool(
                    foundation_context.config.security.maintenance_command_author_ids
                ),
                "guild_allow_list_configured": bool(
                    foundation_context.config.security.discord_guild_allow_list
                ),
            }
            summary = "permissions"
            filename = "kumc-agent-permissions.json"
        else:
            payload = await asyncio.to_thread(automation_context.readiness.cost_report)
            summary = "cost report"
            filename = "kumc-agent-cost-report.json"
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if len(text) <= 1800:
            await interaction.followup.send(f"```json\n{text}\n```", ephemeral=True)
        else:
            await interaction.followup.send(
                content=summary,
                file=_format_markdown_attachment(
                    content=text,
                    filename=filename,
                ),
                ephemeral=True,
            )

    bot.tree.add_command(admin)

    @bot.tree.command(name="ask", description="KUMC-Agent integrated question answering")
    @app_commands.describe(
        question="質問",
        source="検索対象 source",
        mode="回答モード",
        depth="検索深度",
    )
    @app_commands.choices(
        source=[
            app_commands.Choice(name="all", value="all"),
            app_commands.Choice(name="drive", value="drive"),
            app_commands.Choice(name="discord", value="discord"),
            app_commands.Choice(name="notion", value="notion"),
            app_commands.Choice(name="hatena", value="hatena"),
            app_commands.Choice(name="x", value="x"),
            app_commands.Choice(name="crafters_colony", value="crafters_colony"),
            app_commands.Choice(name="minecraft_wiki", value="minecraft_wiki"),
            app_commands.Choice(name="image", value="image"),
            app_commands.Choice(name="member", value="member"),
            app_commands.Choice(name="task", value="task"),
            app_commands.Choice(name="event", value="event"),
            app_commands.Choice(name="server", value="server"),
        ],
        mode=[
            app_commands.Choice(name="answer", value="answer"),
            app_commands.Choice(name="search_only", value="search_only"),
            app_commands.Choice(name="fast", value="fast"),
            app_commands.Choice(name="careful", value="careful"),
        ],
        depth=[
            app_commands.Choice(name="light", value="light"),
            app_commands.Choice(name="normal", value="normal"),
            app_commands.Choice(name="deep", value="deep"),
        ],
    )
    async def ask(
        interaction: discord.Interaction,
        question: str,
        source: str = "all",
        mode: str = "answer",
        depth: str = "normal",
    ) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        if integrated_input_context is None:
            await interaction.followup.send("統合入力受付が未設定です。", ephemeral=True)
            return
        access = _access_context(interaction)
        response = await asyncio.to_thread(
            integrated_input_context.integrated_input.execute,
            IntegratedInputRequest(
                text=question,
                source=source,
                mode=mode,
                depth=depth,
                user_id=access.user_id,
                guild_id=access.guild_id,
                role_ids=access.role_ids,
                is_admin=access.is_admin,
                access=access,
                frontend="discord",
                metadata={"interaction_id": str(interaction.id)},
            ),
        )
        await _send_integrated_response(interaction, response)

    @bot.tree.command(name="work", description="KUMC-Agent workflow operations")
    @app_commands.describe(
        type="Workflow type",
        instruction="指示または本文",
        target="対象 ID / 検索クエリ / 追加本文",
        format="出力形式",
    )
    @app_commands.choices(
        format=[
            app_commands.Choice(name="compact", value="compact"),
            app_commands.Choice(name="markdown", value="markdown"),
        ],
    )
    async def work(
        interaction: discord.Interaction,
        type: str,
        instruction: str = "",
        target: str = "",
        format: app_commands.Choice[str] | None = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        response = await asyncio.to_thread(
            workflow_context.workflow.run,
            WorkRequest(
                work_type=type,
                instruction=instruction,
                target=target,
                output_format=(format.value if format else "markdown"),
                access=_access_context(interaction),
            ),
        )
        kwargs = {"content": response.text, "ephemeral": True}
        if response.detail_markdown and len(response.detail_markdown) > len(response.text):
            kwargs["file"] = _format_markdown_attachment(
                content=response.detail_markdown,
                filename="kumc-agent-work-detail.md",
            )
        task_targets = tuple(getattr(response, "task_candidates", ()) or ()) + tuple(
            getattr(response, "task_change_candidates", ()) or ()
        )
        if task_targets:
            kwargs["view"] = _task_approval_view(task_targets[0].id)
        event_targets = tuple(getattr(response, "event_candidates", ()) or ()) + tuple(
            getattr(response, "event_change_candidates", ()) or ()
        )
        if event_targets:
            kwargs["view"] = _event_approval_view(event_targets[0].id)
        await interaction.followup.send(**kwargs)

    @bot.tree.command(name="approval", description="KUMC-Agent approval operations")
    @app_commands.describe(
        action="Approval action",
        type="Approval target type",
        target_id="Candidate ID",
        comment="コメント",
    )
    @app_commands.choices(
        action=[
            app_commands.Choice(name="list", value="list"),
            app_commands.Choice(name="show", value="show"),
            app_commands.Choice(name="approve", value="approve"),
            app_commands.Choice(name="reject", value="reject"),
            app_commands.Choice(name="edit", value="edit"),
        ],
        type=[
            app_commands.Choice(name="task", value="task"),
            app_commands.Choice(name="event", value="event"),
            app_commands.Choice(name="schedule", value="schedule"),
            app_commands.Choice(name="announcement", value="announcement"),
            app_commands.Choice(name="automation_rule", value="automation_rule"),
            app_commands.Choice(name="server_operation", value="server_operation"),
            app_commands.Choice(name="finance_record", value="finance_record"),
            app_commands.Choice(name="member_assignment", value="member_assignment"),
            app_commands.Choice(name="other", value="other"),
        ],
    )
    async def approval(
        interaction: discord.Interaction,
        action: app_commands.Choice[str],
        type: app_commands.Choice[str],
        target_id: str = "",
        comment: str = "",
    ) -> None:
        if action.value in {"show", "approve", "reject", "edit"} and not target_id:
            await interaction.response.send_message("target_id が必要です。", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        response = await asyncio.to_thread(
            workflow_context.workflow.approval,
            action=action.value,
            target_type=type.value,
            target_id=target_id,
            comment=comment,
            access=_access_context(interaction),
        )
        kwargs = {"content": response.text, "ephemeral": True}
        if response.detail_markdown and len(response.detail_markdown) > len(response.text):
            kwargs["file"] = _format_markdown_attachment(
                content=response.detail_markdown,
                filename="kumc-agent-approval-detail.md",
            )
        task_targets = tuple(getattr(response, "task_candidates", ()) or ()) + tuple(
            getattr(response, "task_change_candidates", ()) or ()
        )
        if task_targets and action.value in {"list", "show", "edit"}:
            kwargs["view"] = _task_approval_view(task_targets[0].id)
        event_targets = tuple(getattr(response, "event_candidates", ()) or ()) + tuple(
            getattr(response, "event_change_candidates", ()) or ()
        )
        if event_targets and type.value == "event" and action.value in {"list", "show", "edit"}:
            kwargs["view"] = _event_approval_view(event_targets[0].id)
        await interaction.followup.send(**kwargs)

    @bot.tree.command(name="automation", description="KUMC-Agent automation operations")
    @app_commands.describe(
        action="Automation action",
        rule_id="Rule ID",
        mode="New mode for set_mode",
        trigger_key="Manual trigger key",
        idempotency_key="Idempotency key",
    )
    @app_commands.choices(
        action=[
            app_commands.Choice(name="list", value="list"),
            app_commands.Choice(name="show", value="show"),
            app_commands.Choice(name="dry_run", value="dry_run"),
            app_commands.Choice(name="run", value="run"),
            app_commands.Choice(name="enable", value="enable"),
            app_commands.Choice(name="disable", value="disable"),
            app_commands.Choice(name="set_mode", value="set_mode"),
        ],
        mode=[
            app_commands.Choice(name="dry_run", value="dry_run"),
            app_commands.Choice(name="approval_required", value="approval_required"),
            app_commands.Choice(name="auto_run", value="auto_run"),
        ],
    )
    async def automation(
        interaction: discord.Interaction,
        action: app_commands.Choice[str],
        rule_id: str = "",
        mode: app_commands.Choice[str] | None = None,
        trigger_key: str = "manual",
        idempotency_key: str = "",
    ) -> None:
        if not _is_authorized(interaction):
            await interaction.response.send_message("権限がありません。", ephemeral=True)
            return
        if action.value != "list" and not rule_id:
            await interaction.response.send_message("rule_id が必要です。", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        access = _access_context(interaction)
        if action.value == "list":
            response = await asyncio.to_thread(automation_context.automation.list_rules)
        elif action.value == "show":
            response = await asyncio.to_thread(
                automation_context.automation.show,
                rule_id=rule_id,
            )
        elif action.value == "enable":
            response = await asyncio.to_thread(
                automation_context.automation.enable,
                rule_id=rule_id,
                access=access,
            )
        elif action.value == "disable":
            response = await asyncio.to_thread(
                automation_context.automation.disable,
                rule_id=rule_id,
                access=access,
            )
        elif action.value == "set_mode":
            response = await asyncio.to_thread(
                automation_context.automation.set_mode,
                rule_id=rule_id,
                mode=mode.value if mode else "dry_run",
                access=access,
            )
        elif action.value == "dry_run":
            response = await asyncio.to_thread(
                automation_context.automation.dry_run,
                rule_id=rule_id,
                trigger_key=trigger_key,
                idempotency_key=idempotency_key,
                access=access,
            )
        else:
            response = await asyncio.to_thread(
                automation_context.automation.run,
                rule_id=rule_id,
                trigger_key=trigger_key,
                idempotency_key=idempotency_key,
                access=access,
            )
        kwargs = {"content": response.text, "ephemeral": True}
        if response.detail_markdown and len(response.detail_markdown) > len(response.text):
            kwargs["file"] = _format_markdown_attachment(
                content=response.detail_markdown,
                filename="kumc-agent-automation-detail.md",
            )
        await interaction.followup.send(**kwargs)

    @bot.event
    async def on_ready() -> None:
        allowed_guilds = foundation_context.config.security.discord_guild_allow_list
        if allowed_guilds:
            for guild_id in allowed_guilds:
                await bot.tree.sync(guild=discord.Object(id=int(guild_id)))
        else:
            await bot.tree.sync()
        logger.info("Discord bot app ready. user=%s", bot.user)

    return bot


__all__ = ["create_bot"]
