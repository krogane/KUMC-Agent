from __future__ import annotations

from pathlib import Path

from kumc_agent.apps.foundation import build_foundation_app_context


def create_app(*, base_dir: Path | None = None):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - depends on deployment env
        raise RuntimeError("fastapi is required to run the API app.") from exc

    context = build_foundation_app_context(base_dir=base_dir)
    app = FastAPI(title="KUMC-Agent API", version="0.2.0")

    @app.get("/health")
    def health() -> dict[str, object]:
        return context.health.check(actor_id="api", actor_type="service").as_dict()

    @app.post("/admin/action/health")
    def admin_health() -> dict[str, object]:
        report = context.health.check(actor_id="api-admin", actor_type="service")
        if report.status == "unhealthy":
            raise HTTPException(status_code=503, detail=report.as_dict())
        return report.as_dict()

    return app


def main(*, host: str = "127.0.0.1", port: int = 8000, base_dir: Path | None = None) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on deployment env
        raise RuntimeError("uvicorn is required to run the API app.") from exc

    uvicorn.run(create_app(base_dir=base_dir), host=host, port=port)
