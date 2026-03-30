from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Literal, Protocol

from kumc_agent.config.schema import RuntimeConfig
from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.ports.embedders import EmbedderPort
from kumc_agent.domain.ports.llms import LLMPort
from kumc_agent.usecases.chat.route import ChatRouteUsecase, RouteRequest

logger = logging.getLogger(__name__)

WarmupStatus = Literal["completed", "skipped", "failed"]


class RerankerPort(Protocol):
    def rerank(self, query: str, chunks: list[Chunk], *, top_k: int) -> list[Chunk]:
        ...


@dataclass(frozen=True)
class WarmupRequest:
    trigger: str = "manual"


@dataclass(frozen=True)
class WarmupStep:
    name: str
    status: WarmupStatus
    detail: str = ""


@dataclass(frozen=True)
class WarmupResult:
    trigger: str
    steps: tuple[WarmupStep, ...]

    @property
    def completed(self) -> int:
        return sum(1 for step in self.steps if step.status == "completed")

    @property
    def skipped(self) -> int:
        return sum(1 for step in self.steps if step.status == "skipped")

    @property
    def failed(self) -> int:
        return sum(1 for step in self.steps if step.status == "failed")


@dataclass(frozen=True)
class _RoutingWarmupTask:
    provider: str
    llama_model_path: str


class WarmupUsecase:
    def __init__(
        self,
        *,
        config: RuntimeConfig,
        embedder: EmbedderPort,
        reranker: RerankerPort | None,
        route_usecase: ChatRouteUsecase,
        rag_llm: LLMPort,
        no_rag_llm: LLMPort,
        refusal_llm: LLMPort,
    ) -> None:
        self._config = config
        self._embedder = embedder
        self._reranker = reranker
        self._route_usecase = route_usecase
        self._rag_llm = rag_llm
        self._no_rag_llm = no_rag_llm
        self._refusal_llm = refusal_llm

    def execute(self, request: WarmupRequest) -> WarmupResult:
        trigger = str(request.trigger or "manual").strip() or "manual"
        logger.info("Warmup started. trigger=%s", trigger)
        steps: list[WarmupStep] = []

        self._execute_step(
            steps=steps,
            name="embedding",
            enabled=self._is_local_embedding_enabled(),
            skip_detail=self._embedding_skip_detail(),
            action=self._warmup_embedding,
        )
        self._execute_step(
            steps=steps,
            name="cross_encoder_reranker",
            enabled=self._is_reranker_local_enabled(),
            skip_detail=self._reranker_skip_detail(),
            action=self._warmup_reranker,
        )
        self._execute_step(
            steps=steps,
            name="routing_function_calling",
            enabled=self._is_local_routing_enabled(),
            skip_detail=self._routing_skip_detail(),
            action=self._warmup_routing,
        )
        self._execute_generation_step(
            steps=steps,
            name="answer_llm",
            provider=self._config.rag.generation.rag.provider,
            model_path=self._config.rag.generation.rag.llama_model_path,
            llm=self._rag_llm,
            temperature=self._config.rag.generation.rag.temperature,
            max_output_tokens=self._config.rag.generation.rag.max_output_tokens,
        )
        self._execute_generation_step(
            steps=steps,
            name="no_rag_llm",
            provider=self._config.rag.generation.no_rag.provider,
            model_path=self._config.rag.generation.no_rag.llama_model_path,
            llm=self._no_rag_llm,
            temperature=self._config.rag.generation.no_rag.temperature,
            max_output_tokens=self._config.rag.generation.no_rag.max_output_tokens,
        )
        self._execute_generation_step(
            steps=steps,
            name="refusal_llm",
            provider=self._config.rag.generation.refusal.provider,
            model_path=self._config.rag.generation.refusal.llama_model_path,
            llm=self._refusal_llm,
            temperature=self._config.rag.generation.refusal.temperature,
            max_output_tokens=self._config.rag.generation.refusal.max_output_tokens,
        )

        result = WarmupResult(trigger=trigger, steps=tuple(steps))
        logger.info(
            "Warmup finished. trigger=%s completed=%s skipped=%s failed=%s",
            trigger,
            result.completed,
            result.skipped,
            result.failed,
        )
        return result

    def _execute_generation_step(
        self,
        *,
        steps: list[WarmupStep],
        name: str,
        provider: str,
        model_path: str,
        llm: LLMPort,
        temperature: float,
        max_output_tokens: int,
    ) -> None:
        if not self._is_local_llm_provider(provider):
            steps.append(
                WarmupStep(
                    name=name,
                    status="skipped",
                    detail=f"provider={provider!r} is not local",
                )
            )
            return
        if not str(model_path or "").strip():
            steps.append(
                WarmupStep(
                    name=name,
                    status="skipped",
                    detail="llama_model_path is not set",
                )
            )
            return
        self._execute_step(
            steps=steps,
            name=name,
            enabled=True,
            skip_detail="",
            action=lambda: self._warmup_llm(
                llm=llm,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )

    @staticmethod
    def _is_local_embedding_provider(provider: str) -> bool:
        normalized = str(provider or "").strip().lower().replace(".", "_")
        return normalized == "local"

    @staticmethod
    def _is_local_llm_provider(provider: str) -> bool:
        normalized = str(provider or "").strip().lower().replace(".", "_")
        return normalized in {"llama", "llama_cpp"}

    def _is_reranker_local_enabled(self) -> bool:
        if not self._config.providers.reranker.enabled:
            return False
        if self._reranker is None:
            return False
        return bool(str(self._config.providers.reranker.model or "").strip())

    def _is_local_embedding_enabled(self) -> bool:
        if not self._is_local_embedding_provider(self._config.providers.embeddings.provider):
            return False
        return bool(str(self._config.providers.embeddings.model or "").strip())

    def _is_local_routing_enabled(self) -> bool:
        if not self._config.rag.routing.enabled:
            return False
        for task in self._routing_tasks():
            if not self._is_local_llm_provider(task.provider):
                continue
            if str(task.llama_model_path or "").strip():
                return True
        return False

    def _embedding_skip_detail(self) -> str:
        provider = str(self._config.providers.embeddings.provider or "").strip()
        if not self._is_local_embedding_provider(provider):
            return f"provider={provider!r} is not local"
        model = str(self._config.providers.embeddings.model or "").strip()
        if not model:
            return "embedding model is not set"
        return ""

    def _reranker_skip_detail(self) -> str:
        if not self._config.providers.reranker.enabled:
            return "reranker is disabled"
        model = str(self._config.providers.reranker.model or "").strip()
        if not model:
            return "reranker model is not set"
        if self._reranker is None:
            return "reranker client is unavailable"
        return ""

    def _routing_skip_detail(self) -> str:
        if not self._config.rag.routing.enabled:
            return "routing is disabled"
        local_tasks = [
            task for task in self._routing_tasks() if self._is_local_llm_provider(task.provider)
        ]
        if not local_tasks:
            return "all routing tasks use non-local providers"
        if not any(str(task.llama_model_path or "").strip() for task in local_tasks):
            return "routing llama_model_path is not set for local tasks"
        return ""

    def _routing_tasks(self) -> tuple[_RoutingWarmupTask, ...]:
        routing_config = self._config.rag.routing
        tasks = getattr(routing_config, "tasks", None)
        if tasks is None:
            return (
                _RoutingWarmupTask(
                    provider=str(getattr(routing_config, "provider", "")),
                    llama_model_path=str(getattr(routing_config, "llama_model_path", "")),
                ),
            )

        task_names = (
            "use_additional_memory",
            "additional_queries",
            "material_names",
            "recency_mode",
        )
        resolved: list[_RoutingWarmupTask] = []
        for name in task_names:
            task = getattr(tasks, name, None)
            if task is None:
                continue
            resolved.append(
                _RoutingWarmupTask(
                    provider=str(getattr(task, "provider", "")),
                    llama_model_path=str(getattr(task, "llama_model_path", "")),
                )
            )
        if resolved:
            return tuple(resolved)
        return (
            _RoutingWarmupTask(
                provider=str(getattr(routing_config, "provider", "")),
                llama_model_path=str(getattr(routing_config, "llama_model_path", "")),
            ),
        )

    def _execute_step(
        self,
        *,
        steps: list[WarmupStep],
        name: str,
        enabled: bool,
        skip_detail: str,
        action,
    ) -> None:
        if not enabled:
            detail = skip_detail or "disabled"
            steps.append(WarmupStep(name=name, status="skipped", detail=detail))
            logger.info("Warmup skipped: %s (%s)", name, detail)
            return
        try:
            action()
            steps.append(WarmupStep(name=name, status="completed"))
            logger.info("Warmup completed: %s", name)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            steps.append(WarmupStep(name=name, status="failed", detail=detail))
            logger.exception("Warmup failed: %s", name)

    def _warmup_embedding(self) -> None:
        self._embedder.embed_query("warmup")
        self._embedder.embed_documents(["warmup document"])

    def _warmup_reranker(self) -> None:
        if self._reranker is None:
            return
        chunks = [
            Chunk(
                id="warmup-0",
                document_id="warmup",
                text="warmup document one",
                index=0,
                metadata={},
            ),
            Chunk(
                id="warmup-1",
                document_id="warmup",
                text="warmup document two",
                index=1,
                metadata={},
            ),
        ]
        self._reranker.rerank("warmup", chunks, top_k=2)

    def _warmup_routing(self) -> None:
        self._route_usecase.execute(RouteRequest(query="warmup"))

    def _warmup_llm(
        self,
        *,
        llm: LLMPort,
        temperature: float,
        max_output_tokens: int,
    ) -> None:
        llm.generate(
            system_prompt="You are a warmup assistant.",
            user_prompt="hello",
            temperature=float(max(0.0, temperature)),
            max_output_tokens=self._warmup_max_tokens(max_output_tokens),
        )

    @staticmethod
    def _warmup_max_tokens(value: int) -> int:
        try:
            raw = int(value)
        except (TypeError, ValueError):
            raw = 1
        return max(1, min(8, raw))
