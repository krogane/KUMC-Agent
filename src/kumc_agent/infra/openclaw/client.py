from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_TRACE_FILENAME = "openclaw_trace.jsonl"
_TRACE_PATH_ENV = "KUMC_OPENCLAW_TRACE_LOG_PATH"
_DEBUG_ENV = "DEBUG"
_REDACTED = "***REDACTED***"
_MAX_STRING_LENGTH = 6000
_SENSITIVE_KEY_TOKENS = (
    "token",
    "secret",
    "password",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
)
_STRING_REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[\w.\-]+@[\w.\-]+\.\w+\b"), "<redacted-email>"),
    (re.compile(r"sk-[A-Za-z0-9]{16,}"), "<redacted-openai-key>"),
    (re.compile(r"AIza[0-9A-Za-z\-_]{16,}"), "<redacted-google-key>"),
    (
        re.compile(r"\b[\w-]{20,}\.[\w-]{6,}\.[\w-]{20,}\b"),
        "<redacted-discord-token>",
    ),
    (
        re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-~+/]+=*"),
        r"\1 <redacted-token>",
    ),
    (
        re.compile(
            r'(?i)"((?:access_)?token|api[_-]?key|secret|password|authorization|cookie|credential)"\s*:\s*"[^"]*"'
        ),
        r'"\1":"***REDACTED***"',
    ),
)


def _is_truthy(value: str | None) -> bool:
    token = str(value or "").strip().lower()
    return token in {"1", "true", "yes", "on"}


def _truncate(text: str, *, limit: int = _MAX_STRING_LENGTH) -> str:
    if len(text) <= limit:
        return text
    remaining = len(text) - limit
    return f"{text[:limit]}...(truncated {remaining} chars)"


def _mask_string(text: str) -> str:
    masked = text
    for pattern, replacement in _STRING_REDACTION_PATTERNS:
        masked = pattern.sub(replacement, masked)
    return _truncate(masked)


def _is_sensitive_key(key: str) -> bool:
    lowered = str(key or "").strip().lower()
    return any(token in lowered for token in _SENSITIVE_KEY_TOKENS)


def _sanitize_for_trace(value: Any, *, key_hint: str | None = None) -> Any:
    if key_hint and _is_sensitive_key(key_hint):
        return _REDACTED
    if isinstance(value, str):
        return _mask_string(value)
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_for_trace(item, key_hint=str(key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_trace(item) for item in value]
    return _truncate(repr(value))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class _OpenClawTraceLogger:
    def __init__(self, *, enabled: bool, file_path: Path) -> None:
        self._enabled = bool(enabled)
        self._file_path = Path(file_path)
        self._lock = threading.Lock()
        self._warned = False

    @classmethod
    def from_env(cls) -> "_OpenClawTraceLogger":
        enabled = _is_truthy(os.getenv(_DEBUG_ENV))
        raw_path = str(os.getenv(_TRACE_PATH_ENV) or "").strip()
        if raw_path:
            file_path = Path(raw_path).expanduser()
        else:
            file_path = Path.cwd() / "logs" / _TRACE_FILENAME
        return cls(enabled=enabled, file_path=file_path)

    def write(self, event: str, **fields: Any) -> None:
        if not self._enabled:
            return
        record: dict[str, Any] = {"ts": _now_iso(), "event": str(event)}
        for key, value in fields.items():
            record[str(key)] = _sanitize_for_trace(value, key_hint=str(key))
        payload = json.dumps(record, ensure_ascii=False)
        try:
            with self._lock:
                self._file_path.parent.mkdir(parents=True, exist_ok=True)
                with self._file_path.open("a", encoding="utf-8") as fp:
                    fp.write(payload)
                    fp.write("\n")
        except OSError as exc:
            if self._warned:
                return
            self._warned = True
            logger.warning(
                "Failed to append OpenClaw trace log. path=%s error=%s",
                self._file_path,
                exc,
            )


@dataclass(frozen=True)
class OpenClawFailure:
    reason: str
    detail: str = ""
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class OpenClawTurnResult:
    text: str
    payload: dict[str, object]


@dataclass(frozen=True)
class OpenClawResponse:
    ok: bool
    result: OpenClawTurnResult | None = None
    failure: OpenClawFailure | None = None


class OpenClawClient:
    def __init__(
        self,
        *,
        enabled: bool,
        agent: str,
        model: str = "",
        config_dir: Path | str | None = None,
        timeout_seconds: float = 120.0,
        command: str = "openclaw",
    ) -> None:
        self._enabled = bool(enabled)
        self._agent = str(agent or "").strip() or "main"
        self._model = self._normalize_model(model)
        self._config_dir = self._resolve_config_dir(config_dir)
        self._workspace_cache: dict[str, Path] = {}
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._command = self._resolve_command(command)
        self._trace = _OpenClawTraceLogger.from_env()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def agent(self) -> str:
        return self._agent

    @staticmethod
    def _normalize_model(model: str) -> str:
        raw = str(model or "").strip()
        lowered = raw.lower()
        if lowered.startswith("gemini/"):
            return f"google/{raw.split('/', 1)[1]}"
        if raw and "/" not in raw and lowered.startswith("gemini-"):
            return f"google/{raw}"
        return raw

    @staticmethod
    def _resolve_config_dir(config_dir: Path | str | None) -> Path | None:
        raw = str(config_dir or "").strip()
        if not raw:
            return None
        resolved = Path(raw).expanduser()
        if not resolved.exists():
            logger.warning(
                "Configured OpenClaw config directory does not exist. Falling back to current working directory. path=%s",
                resolved,
            )
            return None
        if not resolved.is_dir():
            logger.warning(
                "Configured OpenClaw config directory is not a directory. Falling back to current working directory. path=%s",
                resolved,
            )
            return None
        return resolved.resolve()

    def run_turn(
        self,
        *,
        query: str,
        session_id: str,
        user_context: dict[str, object] | None = None,
    ) -> OpenClawResponse:
        if not self._enabled:
            self._trace.write("run_turn_skipped", reason="disabled")
            return OpenClawResponse(ok=False, failure=OpenClawFailure(reason="disabled"))

        cleaned_query = str(query or "").strip()
        if not cleaned_query:
            self._trace.write("run_turn_skipped", reason="empty_query")
            return OpenClawResponse(
                ok=False,
                failure=OpenClawFailure(reason="empty_query"),
            )

        normalized_session = str(session_id or "").strip() or "default"
        self._sync_bootstrap_files(agent=self._agent)
        model_failure = self._configure_model_if_needed()
        if model_failure is not None:
            self._trace.write(
                "run_turn_failed",
                reason=model_failure.reason,
                detail=model_failure.detail,
                return_code=model_failure.return_code,
                stdout=model_failure.stdout,
                stderr=model_failure.stderr,
            )
            return OpenClawResponse(ok=False, failure=model_failure)

        self._trace.write(
            "run_turn_start",
            session_id=normalized_session,
            configured_agent=self._agent,
            query=cleaned_query,
            user_context=user_context or {},
            config_dir=str(self._config_dir) if self._config_dir is not None else "",
        )
        message = self._build_message(
            query=cleaned_query,
            session_id=normalized_session,
            user_context=user_context or {},
        )
        self._trace.write("message_built", message=message)

        active_agent = self._agent
        local_mode = False
        cmd = self._build_command(
            session_id=normalized_session,
            message=message,
            agent=active_agent,
            local=local_mode,
        )
        self._trace.write("command_prepared", command=cmd, local_mode=local_mode)
        completed, failure = self._run_command(cmd)
        if failure is not None:
            self._trace.write(
                "command_failed",
                reason=failure.reason,
                detail=failure.detail,
                return_code=failure.return_code,
                stdout=failure.stdout,
                stderr=failure.stderr,
            )
            return OpenClawResponse(
                ok=False,
                failure=failure,
            )
        assert completed is not None

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        self._trace.write(
            "command_completed",
            return_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            local_mode=local_mode,
            active_agent=active_agent,
        )

        if (
            completed.returncode != 0
            and active_agent
            and self._is_unknown_agent_error(stdout=stdout, stderr=stderr)
        ):
            logger.warning(
                "Configured OpenClaw agent '%s' was not found. Retrying with OpenClaw default agent.",
                active_agent,
            )
            self._trace.write(
                "retry_unknown_agent",
                previous_agent=active_agent,
                return_code=completed.returncode,
                stderr=stderr,
            )
            active_agent = ""
            self._sync_bootstrap_files(agent=active_agent)
            retry_cmd = self._build_command(
                session_id=normalized_session,
                message=message,
                agent=active_agent,
                local=local_mode,
            )
            self._trace.write("command_prepared", command=retry_cmd, local_mode=local_mode)
            retried, retry_failure = self._run_command(retry_cmd)
            if retry_failure is not None:
                self._trace.write(
                    "command_failed",
                    reason=retry_failure.reason,
                    detail=retry_failure.detail,
                    return_code=retry_failure.return_code,
                    stdout=retry_failure.stdout,
                    stderr=retry_failure.stderr,
                )
                return OpenClawResponse(ok=False, failure=retry_failure)
            assert retried is not None
            completed = retried
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            self._trace.write(
                "command_completed",
                return_code=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                local_mode=local_mode,
                active_agent=active_agent,
            )

        if (
            not local_mode
            and self._needs_gateway_retry(
                return_code=completed.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        ):
            logger.warning(
                "OpenClaw gateway is unavailable. Retrying with embedded local mode (--local)."
            )
            self._trace.write(
                "retry_gateway_unavailable",
                return_code=completed.returncode,
                stderr=stderr,
            )
            local_mode = True
            retry_cmd = self._build_command(
                session_id=normalized_session,
                message=message,
                agent=active_agent,
                local=local_mode,
            )
            self._trace.write("command_prepared", command=retry_cmd, local_mode=local_mode)
            retried, retry_failure = self._run_command(retry_cmd)
            if retry_failure is not None:
                self._trace.write(
                    "command_failed",
                    reason=retry_failure.reason,
                    detail=retry_failure.detail,
                    return_code=retry_failure.return_code,
                    stdout=retry_failure.stdout,
                    stderr=retry_failure.stderr,
                )
                return OpenClawResponse(ok=False, failure=retry_failure)
            assert retried is not None
            completed = retried
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            self._trace.write(
                "command_completed",
                return_code=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                local_mode=local_mode,
                active_agent=active_agent,
            )

        if completed.returncode != 0 and self._is_lock_error(stdout=stdout, stderr=stderr):
            logger.warning("OpenClaw session lock detected. Retrying once after short backoff.")
            self._trace.write(
                "retry_session_lock",
                return_code=completed.returncode,
                stderr=stderr,
            )
            time.sleep(1.0)
            retry_cmd = self._build_command(
                session_id=normalized_session,
                message=message,
                agent=active_agent,
                local=local_mode,
            )
            self._trace.write("command_prepared", command=retry_cmd, local_mode=local_mode)
            retried, retry_failure = self._run_command(retry_cmd)
            if retry_failure is not None:
                self._trace.write(
                    "command_failed",
                    reason=retry_failure.reason,
                    detail=retry_failure.detail,
                    return_code=retry_failure.return_code,
                    stdout=retry_failure.stdout,
                    stderr=retry_failure.stderr,
                )
                return OpenClawResponse(ok=False, failure=retry_failure)
            assert retried is not None
            completed = retried
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            self._trace.write(
                "command_completed",
                return_code=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                local_mode=local_mode,
                active_agent=active_agent,
            )

        if completed.returncode != 0:
            self._trace.write(
                "run_turn_failed",
                reason="non_zero_exit",
                return_code=completed.returncode,
                stdout=stdout,
                stderr=stderr,
            )
            return OpenClawResponse(
                ok=False,
                failure=OpenClawFailure(
                    reason="non_zero_exit",
                    detail=f"openclaw exited with code {completed.returncode}",
                    return_code=completed.returncode,
                    stdout=stdout,
                    stderr=stderr,
                ),
            )

        payload = self._extract_json(stdout)
        if payload is None:
            self._trace.write(
                "run_turn_failed",
                reason="invalid_json",
                stdout=stdout,
                stderr=stderr,
            )
            return OpenClawResponse(
                ok=False,
                failure=OpenClawFailure(
                    reason="invalid_json",
                    detail="unable to parse JSON response from openclaw",
                    return_code=completed.returncode,
                    stdout=stdout,
                    stderr=stderr,
                ),
            )

        payload = self._normalize_payload(payload)
        text = self._extract_text(payload)
        if not text:
            self._trace.write(
                "run_turn_failed",
                reason="empty_response",
                payload=payload,
                stderr=stderr,
            )
            return OpenClawResponse(
                ok=False,
                failure=OpenClawFailure(
                    reason="empty_response",
                    detail="openclaw response did not contain answer text",
                    return_code=completed.returncode,
                    stdout=stdout,
                    stderr=stderr,
                ),
            )

        thought_trace = self._extract_thought_trace(payload)
        self._trace.write(
            "run_turn_success",
            route=payload.get("route"),
            response_text=text,
            thought_trace=thought_trace,
            payload=payload,
        )
        return OpenClawResponse(ok=True, result=OpenClawTurnResult(text=text, payload=payload))

    @staticmethod
    def _resolve_command(command: str) -> str:
        cleaned = str(command or "").strip() or "openclaw"
        explicit = Path(cleaned).expanduser()
        if explicit.exists():
            return str(explicit)
        if shutil.which(cleaned):
            return cleaned
        if cleaned == "openclaw":
            candidates = (
                Path(sys.executable).resolve().parent / "openclaw",
                Path.cwd() / "app" / ".venv" / "bin" / "openclaw",
            )
            for candidate in candidates:
                if candidate.exists():
                    return str(candidate)
        return cleaned

    def _build_command(
        self,
        *,
        session_id: str,
        message: str,
        agent: str,
        local: bool = False,
    ) -> list[str]:
        cmd = [
            self._command,
            "agent",
            "--session-id",
            session_id,
            "--message",
            message,
            "--json",
        ]
        if agent:
            cmd[2:2] = ["--agent", agent]
        if local:
            cmd.insert(2, "--local")
        return cmd

    def _configure_model_if_needed(self) -> OpenClawFailure | None:
        if not self._model:
            return None
        cmd = [self._command, "models", "set", self._model]
        self._trace.write("model_configuration_start", model=self._model, command=cmd)
        completed, failure = self._run_command(cmd)
        if failure is not None:
            self._trace.write(
                "model_configuration_failed",
                model=self._model,
                reason=failure.reason,
                detail=failure.detail,
                return_code=failure.return_code,
                stdout=failure.stdout,
                stderr=failure.stderr,
            )
            return OpenClawFailure(
                reason="model_configuration_failed",
                detail=f"{failure.reason}: {failure.detail}".strip(": "),
                return_code=failure.return_code,
                stdout=failure.stdout,
                stderr=failure.stderr,
            )
        assert completed is not None
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if completed.returncode != 0:
            self._trace.write(
                "model_configuration_failed",
                model=self._model,
                return_code=completed.returncode,
                stdout=stdout,
                stderr=stderr,
            )
            return OpenClawFailure(
                reason="model_configuration_failed",
                detail=f"openclaw models set exited with code {completed.returncode}",
                return_code=completed.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        self._trace.write(
            "model_configuration_success",
            model=self._model,
            return_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
        )
        return None

    def _run_command(self, cmd: list[str]) -> tuple[subprocess.CompletedProcess[str] | None, OpenClawFailure | None]:
        env = dict(os.environ)
        # OpenClaw's Ollama provider expects an API key placeholder even for local usage.
        env.setdefault("OLLAMA_API_KEY", "ollama-local")
        project_root = Path(__file__).resolve().parents[4]
        project_src = project_root / "src"
        env.setdefault("KUMC_AGENT_PROJECT_ROOT", str(project_root))
        env.setdefault("KUMC_AGENT_PROJECT_SRC", str(project_src))
        existing_pythonpath = str(env.get("PYTHONPATH", "")).strip()
        if existing_pythonpath:
            pythonpath_entries = existing_pythonpath.split(os.pathsep)
            if str(project_src) not in pythonpath_entries:
                env["PYTHONPATH"] = os.pathsep.join([str(project_src), *pythonpath_entries])
        else:
            env["PYTHONPATH"] = str(project_src)
        gemini_api_key = str(env.get("GEMINI_API_KEY", "")).strip()
        if not gemini_api_key:
            kumc_gemini_api_key = str(env.get("KUMC_GEMINI_API_KEY", "")).strip()
            if kumc_gemini_api_key:
                env["GEMINI_API_KEY"] = kumc_gemini_api_key
            else:
                google_api_key = str(env.get("GOOGLE_API_KEY", "")).strip()
                if google_api_key:
                    env["GEMINI_API_KEY"] = google_api_key
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                env=env,
                text=True,
                timeout=self._timeout_seconds,
            )
            return completed, None
        except FileNotFoundError:
            return (
                None,
                OpenClawFailure(
                    reason="command_not_found",
                    detail=f"{self._command!r} was not found",
                ),
            )
        except subprocess.TimeoutExpired as exc:
            return (
                None,
                OpenClawFailure(
                    reason="timeout",
                    detail=f"timeout after {self._timeout_seconds:.1f}s",
                    stdout=str(exc.stdout or ""),
                    stderr=str(exc.stderr or ""),
                ),
            )
        except Exception as exc:
            logger.exception("OpenClaw invocation crashed.")
            return (
                None,
                OpenClawFailure(
                    reason="execution_error",
                    detail=f"{type(exc).__name__}: {exc}",
                ),
            )

    @staticmethod
    def _is_unknown_agent_error(*, stdout: str, stderr: str) -> bool:
        merged = f"{stdout}\n{stderr}".lower()
        return "unknown agent id" in merged

    @staticmethod
    def _is_gateway_unavailable_error(*, stdout: str, stderr: str) -> bool:
        merged = f"{stdout}\n{stderr}".lower()
        hints = (
            "gateway closed",
            "gateway port",
            "gateway unavailable",
            "failed to connect",
            "connect econnrefused",
            "econnrefused",
        )
        return any(hint in merged for hint in hints)

    def _needs_gateway_retry(self, *, return_code: int, stdout: str, stderr: str) -> bool:
        if not self._is_gateway_unavailable_error(stdout=stdout, stderr=stderr):
            return False
        if return_code != 0:
            return True
        payload = self._extract_json(stdout)
        if payload is None:
            return True
        return not bool(self._extract_text(payload))

    @staticmethod
    def _is_lock_error(*, stdout: str, stderr: str) -> bool:
        merged = f"{stdout}\n{stderr}".lower()
        return (
            "session file locked" in merged
            or "sessions.json.lock" in merged
            or (".lock" in merged and "operation not permitted" in merged)
        )

    def _sync_bootstrap_files(self, *, agent: str) -> None:
        if self._config_dir is None:
            return
        try:
            sources = [
                path
                for path in sorted(self._config_dir.iterdir())
                if path.is_file() and path.suffix.lower() == ".md"
            ]
        except OSError as exc:
            logger.warning(
                "Failed to enumerate OpenClaw config directory for bootstrap sync. config_dir=%s error=%s",
                self._config_dir,
                exc,
            )
            return
        if not sources:
            return
        workspace = self._resolve_workspace(agent=agent)
        if workspace is None:
            return
        try:
            workspace.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "Failed to create OpenClaw workspace directory for bootstrap sync. workspace=%s error=%s",
                workspace,
                exc,
            )
            return

        synced_files: list[str] = []
        for source in sources:
            target = workspace / source.name
            try:
                content = source.read_text(encoding="utf-8")
                if target.exists():
                    existing = target.read_text(encoding="utf-8")
                    if existing == content:
                        continue
                target.write_text(content, encoding="utf-8")
                synced_files.append(source.name)
            except OSError as exc:
                logger.warning(
                    "Failed to sync OpenClaw bootstrap file. source=%s target=%s error=%s",
                    source,
                    target,
                    exc,
                )
        if synced_files:
            self._trace.write(
                "bootstrap_files_synced",
                source_dir=str(self._config_dir),
                workspace=str(workspace),
                files=synced_files,
            )

    def _resolve_workspace(self, *, agent: str) -> Path | None:
        cache_key = str(agent or "").strip() or "main"
        cached = self._workspace_cache.get(cache_key)
        if cached is not None:
            return cached

        workspace = self._resolve_workspace_via_agents_list(agent=cache_key)
        if workspace is None:
            fallback = Path.home() / ".openclaw" / "workspace"
            logger.warning(
                "Unable to resolve OpenClaw workspace via agents list. Falling back to default workspace. path=%s",
                fallback,
            )
            workspace = fallback

        resolved = workspace.expanduser()
        self._workspace_cache[cache_key] = resolved
        return resolved

    def _resolve_workspace_via_agents_list(self, *, agent: str) -> Path | None:
        cmd = [self._command, "agents", "list", "--json"]
        completed, failure = self._run_command(cmd)
        if failure is not None:
            self._trace.write(
                "workspace_resolution_failed",
                reason=failure.reason,
                detail=failure.detail,
                return_code=failure.return_code,
                stderr=failure.stderr,
            )
            return None
        if completed is None:
            return None
        if completed.returncode != 0:
            self._trace.write(
                "workspace_resolution_failed",
                reason="non_zero_exit",
                return_code=completed.returncode,
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
            )
            return None

        entries = self._extract_json_list(completed.stdout or "")
        if not entries:
            return None
        target = str(agent or "").strip()
        selected: Mapping[str, object] | None = None
        if target:
            for item in entries:
                if str(item.get("id") or "").strip() == target:
                    selected = item
                    break
        if selected is None:
            for item in entries:
                if bool(item.get("isDefault")):
                    selected = item
                    break
        if selected is None and entries:
            selected = entries[0]
        if selected is None:
            return None
        workspace = str(selected.get("workspace") or "").strip()
        if not workspace:
            return None
        return Path(workspace)

    @staticmethod
    def _build_message(
        *,
        query: str,
        session_id: str,
        user_context: dict[str, object],
    ) -> str:
        payload = {
            "kind": "kumc_user_query",
            "query": query,
            "history_scope": session_id,
            "user_context": user_context,
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _extract_json(stdout: str) -> dict[str, object] | None:
        text = str(stdout or "").strip()
        if not text:
            return None

        candidates = [line.strip() for line in text.splitlines() if line.strip()]
        for candidate in reversed(candidates):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return OpenClawClient._extract_json_object_from_text(text)
        if isinstance(parsed, dict):
            return parsed
        return None

    @staticmethod
    def _extract_json_object_from_text(text: str) -> dict[str, object] | None:
        for candidate in OpenClawClient._iter_json_object_candidates(text):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def _iter_json_object_candidates(text: str) -> list[str]:
        raw = str(text or "")
        out: list[str] = []
        cursor = 0
        length = len(raw)
        while cursor < length:
            start = raw.find("{", cursor)
            if start < 0:
                break
            depth = 0
            in_string = False
            escaped = False
            end = start
            while end < length:
                char = raw[end]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    end += 1
                    continue
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        out.append(raw[start : end + 1].strip())
                        cursor = end + 1
                        break
                end += 1
            else:
                cursor = start + 1
        return out

    @staticmethod
    def _extract_json_list(stdout: str) -> list[Mapping[str, object]]:
        text = str(stdout or "").strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        out: list[Mapping[str, object]] = []
        for item in parsed:
            if isinstance(item, Mapping):
                out.append(item)
        return out

    @staticmethod
    def _extract_text(payload: dict[str, object]) -> str:
        search_keys = (
            "answer",
            "text",
            "replyText",
            "reply",
            "reply_text",
            "finalText",
            "final",
            "final_text",
            "outputText",
            "output_text",
            "message",
            "response",
            "output",
            "content",
            "summary",
        )
        for key in search_keys:
            value = payload.get(key)
            text = OpenClawClient._extract_text_from_value(value)
            if text:
                return text
        payload_text = OpenClawClient._extract_text_from_payloads(payload.get("payloads"))
        if payload_text:
            return payload_text
        for container_key in (
            "result",
            "response",
            "data",
            "assistant",
            "message",
            "output",
            "meta",
            "delivery",
        ):
            nested = payload.get(container_key)
            text = OpenClawClient._extract_text_from_value(nested)
            if text:
                return text
        for list_key in ("payloads", "messages", "choices", "outputs", "items", "candidates", "parts"):
            value = payload.get(list_key)
            if list_key == "payloads":
                text = OpenClawClient._extract_text_from_payloads(value)
            else:
                text = OpenClawClient._extract_text_from_messages(value)
            if text:
                return text
        return ""

    @staticmethod
    def _normalize_payload(payload: dict[str, object]) -> dict[str, object]:
        embedded = OpenClawClient._extract_embedded_payload(payload)
        if embedded is None:
            merged = dict(payload)
        else:
            merged = dict(payload)
            for key in ("text", "answer", "route", "sources", "routing_decision", "fast_mode", "metadata"):
                value = embedded.get(key)
                if value is None:
                    continue
                if OpenClawClient._is_empty_payload_value(merged.get(key)):
                    merged[key] = value
            if OpenClawClient._is_empty_payload_value(merged.get("fast_mode")):
                embedded_fast_mode = embedded.get("fastmode")
                if embedded_fast_mode is not None:
                    merged["fast_mode"] = embedded_fast_mode
        if OpenClawClient._is_empty_payload_value(merged.get("fast_mode")):
            payload_fast_mode = merged.get("fastmode")
            if payload_fast_mode is not None:
                merged["fast_mode"] = payload_fast_mode
        return merged

    @staticmethod
    def _extract_embedded_payload(payload: dict[str, object]) -> dict[str, object] | None:
        payloads = payload.get("payloads")
        if isinstance(payloads, list):
            for item in reversed(payloads):
                if not isinstance(item, dict):
                    continue
                parsed = OpenClawClient._parse_json_object_string(item.get("text"))
                if parsed is not None:
                    return parsed
        for key in ("text", "answer", "response", "output", "message", "content"):
            parsed = OpenClawClient._parse_json_object_string(payload.get(key))
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _parse_json_object_string(value: object) -> dict[str, object] | None:
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        if not stripped:
            return None
        parsed = OpenClawClient._extract_json(stripped)
        if parsed is None:
            return None
        return parsed

    @staticmethod
    def _is_empty_payload_value(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, dict)):
            return len(value) == 0
        return False

    @staticmethod
    def _extract_text_from_value(value: object) -> str:
        if isinstance(value, str):
            parsed = OpenClawClient._parse_json_object_string(value)
            if parsed is not None:
                text = OpenClawClient._extract_text_from_value(parsed)
                if text:
                    return text
            return value.strip()
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                text = OpenClawClient._extract_text_from_value(item)
                if text:
                    parts.append(text)
            if parts:
                return "\n".join(parts).strip()
            return ""
        if not isinstance(value, dict):
            return ""

        preferred_keys = (
            "text",
            "answer",
            "replyText",
            "reply",
            "reply_text",
            "finalText",
            "final",
            "final_text",
            "outputText",
            "output_text",
            "content",
            "message",
            "response",
            "output",
            "result",
            "summary",
        )
        for key in preferred_keys:
            if key not in value:
                continue
            text = OpenClawClient._extract_text_from_value(value.get(key))
            if text:
                return text

        payload_text = OpenClawClient._extract_text_from_payloads(value.get("payloads"))
        if payload_text:
            return payload_text

        return OpenClawClient._extract_text_from_messages(value.get("messages"))

    @staticmethod
    def _extract_text_from_messages(value: object) -> str:
        if not isinstance(value, list):
            return ""
        assistant_first = []
        others = []
        for item in value:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            if role == "assistant":
                assistant_first.append(item)
            else:
                others.append(item)

        for item in list(reversed(assistant_first)) + list(reversed(others)):
            text = OpenClawClient._extract_text_from_value(item)
            if text:
                return text
        return ""

    @staticmethod
    def _extract_text_from_payloads(value: object) -> str:
        if not isinstance(value, list):
            return ""

        chunks: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            media_lines: list[str] = []

            media_url = item.get("mediaUrl")
            if isinstance(media_url, str) and media_url.strip():
                media_lines.append(f"MEDIA:{media_url.strip()}")

            media_urls = item.get("mediaUrls")
            if isinstance(media_urls, list):
                for url in media_urls:
                    if isinstance(url, str) and url.strip():
                        media_lines.append(f"MEDIA:{url.strip()}")

            if text:
                chunks.append(text)
            if media_lines:
                chunks.extend(media_lines)

        return "\n".join(chunks).strip()

    @staticmethod
    def _extract_thought_trace(payload: dict[str, object]) -> dict[str, object]:
        out: dict[str, object] = {}
        sources: list[tuple[str, Mapping[str, object]]] = [("payload", payload)]
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            sources.append(("metadata", metadata))
        result = payload.get("result")
        if isinstance(result, Mapping):
            sources.append(("result", result))

        key_groups: tuple[tuple[str, tuple[str, ...]], ...] = (
            (
                "intent",
                (
                    "intent",
                    "intent_classification",
                    "intent_label",
                    "classified_intent",
                ),
            ),
            (
                "tool_selection",
                (
                    "tool",
                    "tools",
                    "selected_tool",
                    "selected_tools",
                    "tool_selection",
                    "tool_choice",
                    "tool_selection_reason",
                ),
            ),
            (
                "prompt",
                (
                    "prompt",
                    "prompt_text",
                    "system_prompt",
                    "user_prompt",
                    "messages",
                ),
            ),
            (
                "routing",
                (
                    "route",
                    "routing_decision",
                    "decision_reason",
                    "fast_mode",
                ),
            ),
        )

        for group_name, keys in key_groups:
            group_values: dict[str, object] = {}
            for source_name, source in sources:
                for key in keys:
                    if key in source:
                        group_values[f"{source_name}.{key}"] = source[key]
            if group_values:
                out[group_name] = group_values
        return out
