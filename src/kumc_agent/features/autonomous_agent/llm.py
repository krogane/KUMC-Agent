from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
import re
from typing import Any

from kumc_agent.features.autonomous_agent.sanitizer import sanitize_autonomous_payload


@dataclass(frozen=True)
class AutonomousLLMConfig:
    enabled: bool = False
    prompt_name: str = ""
    prompts_dir: Path | None = None
    temperature: float = 0.0
    max_output_tokens: int = 2048
    max_retries: int = 2


def read_prompt(prompts_dir: Path | None, prompt_name: str, *, fallback: str) -> str:
    if prompts_dir is None or not prompt_name:
        return fallback
    path = prompts_dir / f"{prompt_name}.md"
    if not path.exists() and prompt_name.endswith(".md"):
        path = prompts_dir / prompt_name
    if not path.exists():
        return fallback
    return path.read_text(encoding="utf-8")


def llm_generate(
    llm: object,
    *,
    system_prompt: str,
    user_payload: dict[str, Any],
    temperature: float,
    max_output_tokens: int,
) -> str:
    generate = getattr(llm, "generate")
    return str(
        generate(
            system_prompt=system_prompt,
            user_prompt=json.dumps(user_payload, ensure_ascii=False, default=str),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    )


def load_json_object(text: str) -> dict[str, Any] | None:
    stripped = str(text or "").strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if match:
        stripped = match.group(1).strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(stripped[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def dump_value(value: object, *, limit: int = 1200) -> object:
    if is_dataclass(value):
        return sanitize_autonomous_payload(asdict(value))
    if isinstance(value, dict):
        return sanitize_autonomous_payload(value)
    if isinstance(value, (list, tuple)):
        return [dump_value(item, limit=limit) for item in value[:50]]
    return sanitize_autonomous_payload(value)


def string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return tuple()

