from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from langchain_core.embeddings import Embeddings

from kumc_agent.infra.llm.gemini_rate_limit import wait_for_gemini_rate_limit


def _decode_prompt_env_value(value: str) -> str:
    return value.replace("\\n", "\n")


@lru_cache(maxsize=None)
def get_required_prompt_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Missing required prompt environment variable: {name}")
    return _decode_prompt_env_value(value.strip())


def render_prompt_template(template_env_name: str, **kwargs: object) -> str:
    template = get_required_prompt_env(template_env_name)
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        placeholder = exc.args[0]
        raise RuntimeError(
            f"{template_env_name} is missing placeholder value: {placeholder}"
        ) from exc


def get_llm_chunk_system_prompt() -> str:
    return get_required_prompt_env("PROMPT_LLM_CHUNK_SYSTEM_PROMPT")


def get_raptor_summary_system_prompt() -> str:
    return get_required_prompt_env("PROMPT_RAPTOR_SUMMARY_SYSTEM_PROMPT")


def build_proposition_chunk_prompt(*, text: str) -> str:
    return render_prompt_template(
        "PROMPT_PROPOSITION_CHUNK_TEMPLATE",
        text=text,
    )


def build_summery_chunk_prompt(
    *,
    text: str,
    target_characters: int,
    source_type: str | None = None,
    drive_file_path: str | None = None,
) -> str:
    normalized_type = (source_type or "").strip().lower()
    drive_path = (drive_file_path or "").strip()
    drive_path_display = drive_path if drive_path else "不明"

    if normalized_type in {"messages", "discord_message", "x_posts"}:
        return render_prompt_template(
            "PROMPT_SUMMERY_CHUNK_MESSAGES_TEMPLATE",
            target_characters=target_characters,
            text=text,
        )

    if normalized_type == "sheets":
        return render_prompt_template(
            "PROMPT_SUMMERY_CHUNK_SHEETS_TEMPLATE",
            target_characters=target_characters,
            drive_path_display=drive_path_display,
            text=text,
        )

    return render_prompt_template(
        "PROMPT_SUMMERY_CHUNK_DEFAULT_TEMPLATE",
        target_characters=target_characters,
        drive_path_display=drive_path_display,
        text=text,
    )


def build_raptor_summary_prompt(*, text: str, target_tokens: int) -> str:
    return render_prompt_template(
        "PROMPT_RAPTOR_SUMMARY_TEMPLATE",
        target_tokens=target_tokens,
        text=text,
    )


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    raw_data_dir: Path
    first_rec_chunk_dir: Path
    second_rec_chunk_dir: Path
    sparse_second_rec_chunk_dir: Path
    summery_chunk_dir: Path
    prop_chunk_dir: Path
    raptor_chunk_dir: Path
    index_dir: Path
    discord_bot_token: str = ""
    discord_guild_allow_list: tuple[int, ...] = ()
    gemini_api_key: str = ""
    gemini_requests_per_minute: int = 60
    gemini_summary_requests_per_minute: int = 60
    drive_folder_id: str = ""
    google_application_credentials: str = ""
    drive_max_files: int = 0
    crafters_colony_author_url: str = ""
    crafters_colony_max_pages: int = 0
    crafters_colony_max_articles: int = 0
    pdf_ocr_model_path: str = ""
    embedding_model: str = ""
    raptor_embedding_model: str = ""
    first_rec_chunk_size: int = 1024
    first_rec_chunk_overlap: int = 128
    second_rec_enabled: bool = True
    second_rec_chunk_size: int = 512
    second_rec_chunk_overlap: int = 64
    summery_enabled: bool = True
    summery_characters: int = 200
    summery_provider: str = "llama"
    summery_gemini_model: str = "gemini-3-flash-preview"
    summery_llama_model: str = ""
    summery_llama_model_path: str = ""
    summery_llama_ctx_size: int = 4096
    summery_temperature: float = 0.2
    summery_max_output_tokens: int = 1024
    summery_max_retries: int = 2
    summery_batch_size: int = 1
    llm_provider: str = "llama"
    genai_model: str = "gemini-3-flash-preview"
    llama_model_path: str = ""
    llama_ctx_size: int = 4096
    llama_gpu_layers: int = 0
    llama_threads: int = 4
    temperature: float = 0.0
    max_output_tokens: int = 512
    thinking_level: str = "minimal"
    prop_enabled: bool = False
    prop_provider: str = "llama"
    prop_gemini_model: str = "gemini-3-flash-preview"
    prop_llama_model: str = ""
    prop_llama_model_path: str = ""
    prop_llama_ctx_size: int = 4096
    prop_temperature: float = 0.2
    prop_max_output_tokens: int = 4096
    prop_max_retries: int = 2
    raptor_enabled: bool = False
    raptor_cluster_max_tokens: int = 1024
    raptor_summery_max_tokens: int = 256
    raptor_stop_chunk_count: int = 20
    raptor_k_max: int = 8
    raptor_k_selection: str = "elbow"
    raptor_summery_provider: str = "llama"
    raptor_summery_gemini_model: str = "gemini-3-flash-preview"
    raptor_summery_llama_model: str = ""
    raptor_summery_llama_model_path: str = ""
    raptor_summery_llama_ctx_size: int = 4096
    raptor_summery_temperature: float = 0.2
    raptor_summery_max_retries: int = 2
    clear_raw_data: bool = False
    clear_first_rec_chunk_data: bool = False
    clear_second_rec_chunk_data: bool = False
    clear_summery_chunk_data: bool = False
    clear_prop_chunk_data: bool = False
    clear_raptor_chunk_data: bool = False
    update_raw_data: bool = True
    update_first_rec_chunk_data: bool = True
    update_second_rec_chunk_data: bool = True
    update_sparse_second_rec_chunk_data: bool = True
    update_summery_chunk_data: bool = True
    update_prop_chunk_data: bool = True
    update_raptor_chunk_data: bool = True
    sudachi_mode: str = "B"
    sparse_bm25_k1: float = 1.2
    sparse_bm25_b: float = 0.75
    sparse_use_normalized_form: bool = True
    sparse_remove_symbols: bool = True


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, *, model_path: str) -> None:
        if not model_path:
            raise RuntimeError("Embedding model path is required.")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for embedding access."
            ) from exc

        self._model_path = model_path
        self._model = SentenceTransformer(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        self._use_e5_prefix = _is_multilingual_e5(model_path)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._use_e5_prefix:
            texts = [self._apply_e5_prefix(text, prefix="document:") for text in texts]
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return _vectors_to_list(vectors)

    def embed_query(self, text: str) -> list[float]:
        query = text if text else " "
        if self._use_e5_prefix:
            query = self._apply_e5_prefix(query, prefix="query:")
        vectors = self._model.encode([query], normalize_embeddings=True)
        return _vectors_to_list(vectors)[0] if vectors is not None else []

    @staticmethod
    def _apply_e5_prefix(text: str, *, prefix: str) -> str:
        stripped = (text or "").lstrip()
        lower = stripped.lower()
        if lower.startswith("query:") or lower.startswith("document:"):
            return stripped
        if not stripped:
            return f"{prefix} "
        return f"{prefix} {stripped}"


class GeminiEmbeddings(Embeddings):
    _BATCH_SIZE = 96

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None = None,
        requests_per_minute: int | None = None,
    ) -> None:
        self._model_name = (model_name or "").strip()
        if not self._model_name:
            raise RuntimeError("Gemini embedding model name is required.")

        resolved_api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        if not resolved_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Please set it in .env")

        self._api_key = resolved_api_key
        self._requests_per_minute = max(
            0,
            requests_per_minute
            if requests_per_minute is not None
            else int(
                os.getenv(
                    "KUMC_GEMINI_REQUESTS_PER_MINUTE",
                    os.getenv("GEMINI_REQUESTS_PER_MINUTE", "60"),
                )
            ),
        )
        self._client = _gemini_embedding_client(self._api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        normalized_texts = [_normalize_embedding_text(text) for text in texts]
        vectors: list[list[float]] = []
        for i in range(0, len(normalized_texts), self._BATCH_SIZE):
            batch = normalized_texts[i : i + self._BATCH_SIZE]
            wait_for_gemini_rate_limit(
                max_requests_per_minute=self._requests_per_minute
            )
            response = self._client.models.embed_content(
                model=self._model_name,
                contents=batch,
                config=_gemini_embed_config(task_type="RETRIEVAL_DOCUMENT"),
            )
            vectors.extend(_extract_gemini_embedding_vectors(response))
        if len(vectors) != len(normalized_texts):
            raise RuntimeError(
                "Gemini embedding response count mismatch for documents: "
                f"requested={len(normalized_texts)} got={len(vectors)}"
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        wait_for_gemini_rate_limit(
            max_requests_per_minute=self._requests_per_minute
        )
        response = self._client.models.embed_content(
            model=self._model_name,
            contents=[_normalize_embedding_text(text)],
            config=_gemini_embed_config(task_type="RETRIEVAL_QUERY"),
        )
        vectors = _extract_gemini_embedding_vectors(response)
        if not vectors or not vectors[0]:
            raise RuntimeError(
                "Gemini embedding response did not contain a query vector."
            )
        return vectors[0]


class EmbeddingFactory:
    def __init__(self, model_name: str, *, api_key: str | None = None) -> None:
        self._model_name = model_name
        self._api_key = api_key

    @property
    def model_name(self) -> str:
        return self._model_name

    @lru_cache(maxsize=1)
    def get_embeddings(self) -> Embeddings:
        provider, model_name = _parse_embedding_model_spec(self._model_name)
        if provider == "gemini":
            return GeminiEmbeddings(model_name=model_name, api_key=self._api_key)
        return SentenceTransformerEmbeddings(model_path=model_name)


def _vectors_to_list(vectors) -> list[list[float]]:
    tolist = getattr(vectors, "tolist", None)
    if callable(tolist):
        return tolist()
    return [list(vector) for vector in vectors]


def _is_multilingual_e5(model_path: str) -> bool:
    normalized = (model_path or "").lower()
    return "multilingual-e5" in normalized or "multilingual_e5" in normalized


def _normalize_embedding_text(text: str | None) -> str:
    normalized = text if text else " "
    return normalized if normalized.strip() else " "


def _parse_embedding_model_spec(model_name: str) -> tuple[str, str]:
    raw = (model_name or "").strip()
    lowered = raw.lower()
    if lowered.startswith("gemini:"):
        parsed = raw.split(":", maxsplit=1)[1].strip()
        if not parsed:
            raise RuntimeError(
                "Gemini embedding model is missing. "
                "Use EMBEDDING_MODEL=gemini:<model-name>."
            )
        return "gemini", parsed
    if lowered.startswith("gemini/"):
        parsed = raw.split("/", maxsplit=1)[1].strip()
        if not parsed:
            raise RuntimeError(
                "Gemini embedding model is missing. "
                "Use EMBEDDING_MODEL=gemini/<model-name>."
            )
        return "gemini", parsed
    return "local", raw


def _extract_gemini_embedding_vectors(response) -> list[list[float]]:
    embeddings = getattr(response, "embeddings", None) or []
    if not embeddings:
        single = getattr(response, "embedding", None)
        if single is not None:
            embeddings = [single]
    vectors: list[list[float]] = []
    for embedding in embeddings:
        values = getattr(embedding, "values", None) or []
        vectors.append([float(value) for value in values])
    return vectors


def _gemini_embed_config(*, task_type: str):
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is required for Gemini embedding access."
        ) from exc
    return genai.types.EmbedContentConfig(task_type=task_type)


@lru_cache(maxsize=1)
def _gemini_embedding_client(api_key: str):
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is required for Gemini embedding access."
        ) from exc

    return genai.Client(api_key=api_key)
