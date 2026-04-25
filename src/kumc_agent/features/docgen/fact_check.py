from __future__ import annotations

import re

from kumc_agent.domain.models.docgen import FactCheckFinding


_UNSAFE_PATTERNS = (
    ("credential", re.compile(r"(password|token|secret|api[_ -]?key|sk-[A-Za-z0-9_-]{16,})", re.I)),
    ("pin", re.compile(r"\b(PIN|暗証番号)\b[:：]?\s*\d{3,}", re.I)),
    ("internal_ip", re.compile(r"\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})\b")),
    ("network_key", re.compile(r"(wifi|wi-fi|無線|ネットワーク).{0,12}(key|キー|password|パスワード)", re.I)),
    ("unlock_procedure", re.compile(r"(解錠|鍵管理|鍵の場所|入室手順)")),
    ("personal_data", re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b\d{2,4}-\d{2,4}-\d{3,4}\b")),
)


class FactCheckService:
    def inspect(self, text: str, *, public: bool) -> tuple[str, tuple[FactCheckFinding, ...]]:
        findings: list[FactCheckFinding] = []
        sanitized = text
        for kind, pattern in _UNSAFE_PATTERNS:
            for match in pattern.finditer(text):
                severity = "high" if public else "medium"
                findings.append(
                    FactCheckFinding(
                        kind=kind,
                        message=f"{kind} に該当する可能性がある情報を検出しました。",
                        severity=severity,
                    )
                )
            if public:
                sanitized = pattern.sub("[公開不可情報を削除]", sanitized)
        if _needs_fact_check(text):
            findings.append(
                FactCheckFinding(
                    kind="unverified_fact",
                    message="日時・場所・金額・申込条件は公開前に一次資料で確認してください。",
                    severity="medium",
                )
            )
        return sanitized, tuple(findings)


def _needs_fact_check(text: str) -> bool:
    return bool(re.search(r"\d{1,4}[/-]\d{1,2}|月\d{1,2}日|場所|会場|円|無料|申込|参加条件", text))
