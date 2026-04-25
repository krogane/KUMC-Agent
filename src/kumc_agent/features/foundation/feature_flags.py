from __future__ import annotations

from dataclasses import asdict, dataclass

from kumc_agent.config.schema import RiskFeatureFlagsSection


_VALID_MODES = {"disabled", "approval_required", "enabled"}


@dataclass(frozen=True)
class FeatureFlagService:
    risk_flags: RiskFeatureFlagsSection

    def modes(self) -> dict[str, str]:
        return {
            name: self._normalize_mode(value)
            for name, value in asdict(self.risk_flags).items()
        }

    def mode_for(self, name: str) -> str:
        try:
            return self.modes()[name]
        except KeyError as exc:
            raise KeyError(f"Unknown risk feature flag: {name}") from exc

    def is_disabled(self, name: str) -> bool:
        return self.mode_for(name) == "disabled"

    def requires_approval(self, name: str) -> bool:
        return self.mode_for(name) == "approval_required"

    def disabled_flags(self) -> tuple[str, ...]:
        return tuple(name for name, mode in self.modes().items() if mode == "disabled")

    @staticmethod
    def _normalize_mode(value: str) -> str:
        mode = (value or "").strip().lower()
        if mode not in _VALID_MODES:
            raise ValueError(
                "Risk feature mode must be one of: disabled, approval_required, enabled."
            )
        return mode
