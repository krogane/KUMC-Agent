from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Iterable
from pathlib import Path

import numpy as np

from config import AppConfig, RAPTOR_SUMMARY_SYSTEM_PROMPT, build_raptor_summary_prompt
from indexing.chunks import Chunk, load_chunks_from_dirs, write_chunks
from indexing.llm_client import generate_text
from indexing.token_utils import estimate_tokens
from indexing.utils import ensure_dir

logger = logging.getLogger(__name__)


def build_raptor_summaries(
    *,
    chunks: list[Chunk],
    config: AppConfig,
    source_label: str,
) -> list[Chunk]:
    if not chunks:
        return []

    current_chunks = list(chunks)
    all_summaries: list[Chunk] = []
    level = 0

    while current_chunks:
        summaries = _summarize_level(
            chunks=current_chunks,
            config=config,
            source_label=source_label,
            level=level,
        )
        if not summaries:
            break

        all_summaries.extend(summaries)
        if len(summaries) <= config.raptor_stop_chunk_count:
            break
        current_chunks = summaries
        level += 1

    return all_summaries


def raptor_chunk_global(
    *,
    input_chunk_dirs: list[Path],
    output_chunk_dir: Path,
    config: AppConfig,
    output_filename: str = "raptor_global.jsonl",
    skip_existing: bool = False,
) -> None:
    ensure_dir(output_chunk_dir)

    out_path = output_chunk_dir / output_filename
    if skip_existing and out_path.exists():
        latest_input = _latest_input_mtime(input_chunk_dirs)
        if latest_input is None:
            logger.warning("No chunks available for RAPTOR under %s", output_chunk_dir)
            return
        if out_path.stat().st_mtime >= latest_input:
            logger.info("Skip RAPTOR (up-to-date): %s", out_path.name)
            return
        logger.info("Rebuilding RAPTOR (inputs updated): %s", out_path.name)

    chunks = load_chunks_from_dirs(input_chunk_dirs)
    if not chunks:
        logger.warning("No chunks available for RAPTOR under %s", output_chunk_dir)
        return

    summaries = build_raptor_summaries(
        chunks=chunks,
        config=config,
        source_label=output_filename,
    )

    output_chunks: list[Chunk] = []
    for idx, summary in enumerate(summaries):
        metadata = dict(summary.metadata)
        metadata["chunk_id"] = idx
        output_chunks.append(Chunk(text=summary.text, metadata=metadata))

    write_chunks(out_path, output_chunks)
    logger.info(
        "RAPTOR summarized %d chunks -> %s (%d chunks)",
        len(chunks),
        out_path.name,
        len(output_chunks),
    )


def _latest_input_mtime(chunk_dirs: Iterable[Path]) -> float | None:
    latest: float | None = None
    for chunk_dir in chunk_dirs:
        if not chunk_dir.exists():
            continue
        for path in chunk_dir.glob("*.jsonl"):
            try:
                mtime = path.stat().st_mtime
            except FileNotFoundError:
                continue
            if latest is None or mtime > latest:
                latest = mtime
    return latest


def _summarize_level(
    *,
    chunks: list[Chunk],
    config: AppConfig,
    source_label: str,
    level: int,
) -> list[Chunk]:
    embeddings = _embed_chunks(chunks, config=config)
    if embeddings.size == 0:
        return []

    clusters = _cluster_chunks(
        embeddings=embeddings,
        chunks=chunks,
        max_cluster_tokens=config.raptor_cluster_max_tokens,
        k_max=config.raptor_k_max,
        selection_method=config.raptor_k_selection,
        tokenizer_model=config.raptor_embedding_model,
    )
    summaries: list[Chunk] = []

    for cluster_id, indices in enumerate(clusters):
        cluster_chunks = [chunks[i] for i in indices]
        summary = _summarize_cluster(
            cluster_chunks=cluster_chunks,
            config=config,
            source_label=source_label,
            level=level,
            cluster_id=cluster_id,
        )
        if summary:
            summaries.append(summary)

    logger.info(
        "RAPTOR level %d produced %d summaries from %d chunks",
        level,
        len(summaries),
        len(chunks),
    )
    return summaries


def _embed_chunks(chunks: list[Chunk], *, config: AppConfig) -> np.ndarray:
    model_name = config.raptor_embedding_model
    if not chunks:
        return np.array([])
    from config import EmbeddingFactory

    embedder = EmbeddingFactory(model_name).get_embeddings()
    texts = []
    for chunk in chunks:
        drive_path = str(chunk.metadata.get("drive_file_path") or "")
        texts.append(f"{chunk.text}\n{drive_path}".strip())

    vectors = embedder.embed_documents(texts)
    return np.array(vectors, dtype=np.float32)


def _cluster_chunks(
    *,
    embeddings: np.ndarray,
    chunks: list[Chunk],
    max_cluster_tokens: int,
    k_max: int,
    selection_method: str,
    tokenizer_model: str,
) -> list[list[int]]:
    n = len(chunks)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    k_max = max(1, min(k_max, n))
    k = _select_k(embeddings, k_max=k_max, method=selection_method)
    labels = _kmeans_labels(embeddings, k)

    grouped = _group_by_label(labels)
    tokens = [
        estimate_tokens(text=chunk.text, model_name=tokenizer_model)
        for chunk in chunks
    ]

    clusters: list[list[int]] = []
    for indices in grouped:
        clusters.extend(
            _split_cluster_by_token_limit(
                indices=indices,
                embeddings=embeddings,
                tokens=tokens,
                max_tokens=max_cluster_tokens,
            )
        )

    return clusters


def _summarize_cluster(
    *,
    cluster_chunks: list[Chunk],
    config: AppConfig,
    source_label: str,
    level: int,
    cluster_id: int,
) -> Chunk | None:
    include_drive_path = level == 0
    formatted_chunks = [
        _format_chunk_for_prompt(
            chunk=chunk, include_drive_path=include_drive_path
        )
        for chunk in cluster_chunks
    ]
    text = "\n\n".join(chunk_text for chunk_text in formatted_chunks if chunk_text)
    if not text.strip():
        return None

    prompt = build_raptor_summary_prompt(
        text=text,
        target_tokens=config.raptor_summary_max_tokens,
    )

    summary_text = _run_summary_llm(
        prompt=prompt,
        source_name=f"{source_label}#L{level}C{cluster_id}",
        config=config,
    )
    if not summary_text:
        return None

    metadata = _build_summary_metadata(
        cluster_chunks, level=level, cluster_id=cluster_id
    )
    return Chunk(text=summary_text, metadata=metadata)


def _run_summary_llm(*, prompt: str, source_name: str, config: AppConfig) -> str | None:
    last_error: Exception | None = None
    for attempt in range(1, config.raptor_summary_max_retries + 1):
        try:
            response = generate_text(
                provider=config.raptor_summary_provider,
                api_key=config.gemini_api_key,
                prompt=prompt,
                model=config.raptor_summary_model,
                system_prompt=RAPTOR_SUMMARY_SYSTEM_PROMPT,
                llama_model_path=config.raptor_summary_llama_model_path,
                llama_ctx_size=config.raptor_summary_llama_ctx_size,
                temperature=config.raptor_summary_temperature,
                max_output_tokens=config.raptor_summary_max_tokens,
                thinking_level=config.thinking_level,
                llama_threads=config.llama_threads,
                llama_gpu_layers=config.llama_gpu_layers,
            )
            text = response.strip()
            if text:
                return text
        except Exception as exc:
            last_error = exc
            logger.warning(
                "RAPTOR summary failed for %s (attempt %d/%d): %s",
                source_name,
                attempt,
                config.raptor_summary_max_retries,
                exc,
            )

    if last_error:
        logger.error("RAPTOR summary failed for %s: %s", source_name, last_error)
    return None


def _select_k(embeddings: np.ndarray, *, k_max: int, method: str) -> int:
    if embeddings.shape[0] <= 1:
        return 1
    if (method or "").lower() == "silhouette":
        k = _select_k_silhouette(embeddings, k_max=k_max)
        if k:
            return k
        logger.warning("Silhouette selection unavailable, falling back to elbow.")
    return _select_k_elbow(embeddings, k_max=k_max)


def _select_k_elbow(embeddings: np.ndarray, *, k_max: int) -> int:
    ks = list(range(1, k_max + 1))
    inertias: list[float] = []
    for k in ks:
        inertias.append(_kmeans_inertia(embeddings, k))

    if len(ks) <= 2:
        return ks[-1]

    points = np.column_stack((ks, inertias))
    start, end = points[0], points[-1]
    line_vec = end - start
    line_vec_norm = line_vec / (np.linalg.norm(line_vec) + 1e-9)
    distances: list[float] = []
    for point in points:
        vec = point - start
        proj_len = np.dot(vec, line_vec_norm)
        proj = start + proj_len * line_vec_norm
        dist = np.linalg.norm(point - proj)
        distances.append(dist)

    best_index = int(np.argmax(distances))
    return ks[best_index]


def _select_k_silhouette(embeddings: np.ndarray, *, k_max: int) -> int | None:
    try:
        from sklearn.metrics import silhouette_score
    except Exception:
        return None

    best_score = -1.0
    best_k: int | None = None
    for k in range(2, k_max + 1):
        labels = _kmeans_labels(embeddings, k)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def _kmeans_labels(embeddings: np.ndarray, k: int) -> np.ndarray:
    _, labels, _ = _kmeans(embeddings, k)
    return labels


def _kmeans_inertia(embeddings: np.ndarray, k: int) -> float:
    _, _, inertia = _kmeans(embeddings, k)
    return inertia


def _kmeans(
    embeddings: np.ndarray,
    k: int,
    max_iter: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = embeddings.shape[0]
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centroids = embeddings[rng.choice(n, size=k, replace=False)]

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        distances = _pairwise_distances(embeddings, centroids)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for idx in range(k):
            members = embeddings[labels == idx]
            if len(members) == 0:
                centroids[idx] = embeddings[rng.integers(0, n)]
            else:
                centroids[idx] = members.mean(axis=0)

    distances = _pairwise_distances(embeddings, centroids)
    inertia = float(np.sum(np.min(distances, axis=1) ** 2))
    return centroids, labels, inertia


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_sq = np.sum(a * a, axis=1, keepdims=True)
    b_sq = np.sum(b * b, axis=1, keepdims=True).T
    return a_sq + b_sq - 2 * np.dot(a, b.T)


def _group_by_label(labels: np.ndarray) -> list[list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        grouped[label].append(idx)
    return list(grouped.values())


def _split_cluster_by_token_limit(
    *,
    indices: list[int],
    embeddings: np.ndarray,
    tokens: list[int],
    max_tokens: int,
) -> list[list[int]]:
    if not indices:
        return []
    total_tokens = sum(tokens[i] for i in indices)
    if total_tokens <= max_tokens or len(indices) == 1:
        return [indices]

    target_k = min(len(indices), max(2, int(math.ceil(total_tokens / max_tokens))))
    sub_embeddings = embeddings[indices]
    labels = _kmeans_labels(sub_embeddings, target_k)
    groups = _group_by_label(labels)

    clusters: list[list[int]] = []
    for group in groups:
        mapped = [indices[i] for i in group]
        if sum(tokens[i] for i in mapped) > max_tokens and len(mapped) > 1:
            clusters.extend(
                _split_cluster_by_token_limit(
                    indices=mapped,
                    embeddings=embeddings,
                    tokens=tokens,
                    max_tokens=max_tokens,
                )
            )
        else:
            clusters.append(mapped)
    return clusters


def _build_summary_metadata(
    chunks: Iterable[Chunk], *, level: int, cluster_id: int
) -> dict[str, object]:
    def common_or_mixed(values: Iterable[str]) -> str:
        unique = {value for value in values if value}
        if len(unique) == 1:
            return unique.pop()
        if not unique:
            return ""
        return "mixed"

    metadata = {
        "source_file_name": common_or_mixed(
            [str(chunk.metadata.get("source_file_name", "")) for chunk in chunks]
        ),
        "source_type": common_or_mixed(
            [str(chunk.metadata.get("source_type", "")) for chunk in chunks]
        ),
        "drive_file_name": common_or_mixed(
            [str(chunk.metadata.get("drive_file_name", "")) for chunk in chunks]
        ),
        "drive_mime_type": common_or_mixed(
            [str(chunk.metadata.get("drive_mime_type", "")) for chunk in chunks]
        ),
        "drive_file_path": common_or_mixed(
            [str(chunk.metadata.get("drive_file_path", "")) for chunk in chunks]
        ),
        "drive_file_id": common_or_mixed(
            [str(chunk.metadata.get("drive_file_id", "")) for chunk in chunks]
        ),
        "chunk_stage": "raptor",
        "raptor_level": level,
        "raptor_cluster_id": cluster_id,
    }
    return metadata


def _format_chunk_for_prompt(*, chunk: Chunk, include_drive_path: bool) -> str:
    text = chunk.text.strip()
    if not text:
        return ""
    if not include_drive_path:
        return text
    drive_path = str(chunk.metadata.get("drive_file_path") or "")
    if not drive_path:
        return text
    return f"{text}\ndrive_file_path: {drive_path}"
