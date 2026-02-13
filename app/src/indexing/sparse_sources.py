from __future__ import annotations

from pathlib import Path

from config import AppConfig

BASE_SOURCE_DIRS = ("docs", "sheets", "messages", "hatenablog")
SECOND_REC_SOURCE_DIRS = ("docs", "sheets", "messages", "vc", "hatenablog")


def first_rec_chunk_dirs(config: AppConfig) -> list[Path]:
    dirs: list[Path] = []
    for name in BASE_SOURCE_DIRS:
        candidate = config.first_rec_chunk_dir / name
        if candidate.exists():
            dirs.append(candidate)
    return dirs


def second_rec_chunk_dirs(config: AppConfig) -> list[Path]:
    if not config.second_rec_enabled:
        dirs = first_rec_chunk_dirs(config)
        vc_second = config.second_rec_chunk_dir / "vc"
        if vc_second.exists():
            dirs.append(vc_second)
        return dirs

    dirs: list[Path] = []
    for name in SECOND_REC_SOURCE_DIRS:
        candidate = config.second_rec_chunk_dir / name
        if candidate.exists():
            dirs.append(candidate)
    return dirs


def sparse_second_rec_chunk_dirs(config: AppConfig) -> list[Path]:
    if not config.second_rec_enabled:
        dirs = first_rec_chunk_dirs(config)
        vc_sparse = config.sparse_second_rec_chunk_dir / "vc"
        if vc_sparse.exists():
            dirs.append(vc_sparse)
        else:
            vc_second = config.second_rec_chunk_dir / "vc"
            if vc_second.exists():
                dirs.append(vc_second)
        return dirs

    dirs: list[Path] = []
    for name in SECOND_REC_SOURCE_DIRS:
        candidate = config.sparse_second_rec_chunk_dir / name
        if candidate.exists():
            dirs.append(candidate)
    if dirs:
        return dirs
    return second_rec_chunk_dirs(config)


def sparse_chunk_dirs(config: AppConfig) -> list[Path]:
    dirs: list[Path] = []

    if config.second_rec_enabled:
        sparse_second_rec_dirs: list[Path] = []
        for name in SECOND_REC_SOURCE_DIRS:
            candidate = config.sparse_second_rec_chunk_dir / name
            if candidate.exists():
                sparse_second_rec_dirs.append(candidate)
        if sparse_second_rec_dirs:
            dirs.extend(sparse_second_rec_dirs)
        else:
            for name in SECOND_REC_SOURCE_DIRS:
                fallback = config.second_rec_chunk_dir / name
                if fallback.exists():
                    dirs.append(fallback)
    else:
        for name in BASE_SOURCE_DIRS:
            candidate = config.first_rec_chunk_dir / name
            if candidate.exists():
                dirs.append(candidate)
        vc_sparse = config.sparse_second_rec_chunk_dir / "vc"
        if vc_sparse.exists():
            dirs.append(vc_sparse)
        else:
            vc_second = config.second_rec_chunk_dir / "vc"
            if vc_second.exists():
                dirs.append(vc_second)

    if config.prop_enabled and config.second_rec_enabled:
        for name in ("docs", "sheets", "hatenablog"):
            candidate = config.prop_chunk_dir / name
            if candidate.exists():
                dirs.append(candidate)

    if config.raptor_enabled:
        raptor_dir = config.raptor_chunk_dir
        if raptor_dir.exists():
            dirs.append(raptor_dir)
    return dirs
