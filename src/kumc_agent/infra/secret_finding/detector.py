from __future__ import annotations

from dataclasses import dataclass
import re

from kumc_agent.domain.models.secret import SecretFinding
from kumc_agent.utils.hashing import stable_hash


@dataclass(frozen=True)
class SecretRule:
    secret_type: str
    severity: str
    redaction_policy: str
    pattern: re.Pattern[str]


class SecretFindingDetector:
    def __init__(self) -> None:
        self._rules = (
            SecretRule(
                "credential",
                "critical",
                "deny",
                re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
            ),
            SecretRule(
                "credential",
                "critical",
                "deny",
                re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
            ),
            SecretRule(
                "credential",
                "critical",
                "deny",
                re.compile(r"[MN][A-Za-z\d]{23}\.[\w-]{6}\.[\w-]{20,}"),
            ),
            SecretRule(
                "credential",
                "critical",
                "deny",
                re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----"),
            ),
            SecretRule(
                "network_key",
                "high",
                "admin_only",
                re.compile(r"(?i)\b(?:password|passwd|token|secret|api[_-]?key)\s*[:=]\s*[^\s`'\";]{8,}"),
            ),
            SecretRule(
                "internal_ip",
                "medium",
                "summary_only",
                re.compile(r"\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b"),
            ),
            SecretRule(
                "personal_data",
                "medium",
                "summary_only",
                re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
            ),
            SecretRule(
                "personal_data",
                "medium",
                "summary_only",
                re.compile(r"\b0\d{1,4}-\d{1,4}-\d{3,4}\b"),
            ),
            SecretRule(
                "finance",
                "high",
                "admin_only",
                re.compile(r"(?:口座番号|銀行口座|振込先|暗証番号|PIN)\s*[:：]?\s*\S+"),
            ),
        )

    def detect(
        self,
        *,
        source_item_id: str,
        text: str,
        chunk_id: str | None = None,
    ) -> list[SecretFinding]:
        findings: list[SecretFinding] = []
        seen: set[tuple[str, str]] = set()
        for rule in self._rules:
            for match in rule.pattern.finditer(text or ""):
                span = match.group(0)
                span_hash = stable_hash(span)
                key = (rule.secret_type, span_hash)
                if key in seen:
                    continue
                seen.add(key)
                findings.append(
                    SecretFinding(
                        source_item_id=source_item_id,
                        chunk_id=chunk_id,
                        secret_type=rule.secret_type,
                        severity=rule.severity,
                        redaction_policy=rule.redaction_policy,
                        detected_span_hash=span_hash,
                        metadata={"match_start": match.start(), "match_end": match.end()},
                    )
                )
        return findings


def strictest_redaction_policy(findings: list[SecretFinding]) -> str:
    priority = {
        "quote_allowed": 0,
        "summary_only": 1,
        "admin_only": 2,
        "deny": 3,
    }
    mode = "quote_allowed"
    for finding in findings:
        if priority.get(finding.redaction_policy, 0) > priority.get(mode, 0):
            mode = finding.redaction_policy
    return mode
