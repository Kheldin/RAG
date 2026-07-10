"""Indexing and retrieval helpers for the BM25-backed RAG pipeline."""

import os
import re
import bm25s  # type: ignore
from typing import Any, cast
from src.models.models import MinimalSource
from .exceptions import LLMException, LogType, log_message


_PATH_PREFIXES = [
    "data/raw/vllm-0.10.1/",
    "vllm-0.10.1/",
    "./vllm-0.10.1/",
]

_BASE_DIRS = ["", "data/raw/vllm-0.10.1/", "vllm-0.10.1/"]


def normalize_path(path: str) -> str:
    """Normalize a repository path to the relative source path used in datasets."""
    path = path.replace("\\", "/")
    for prefix in _PATH_PREFIXES:
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def get_real_path(normalized_path: str) -> str:
    """Resolve a normalized path to an existing file on disk."""
    for base in _BASE_DIRS:
        candidate = os.path.join(base, normalized_path)
        if os.path.exists(candidate):
            return candidate
    return normalized_path


def _strip_injected_header(text: str) -> str:
    """Remove the synthetic file header added during indexing."""
    return re.sub(r"^--- File: .*? ---\n", "", text)


def locate_character_indices(file_path: str, chunk_text: str) -> tuple[int, int]:
    """Find the approximate character range of chunk_text inside file_path."""
    clean_chunk = _strip_injected_header(chunk_text)
    stripped = clean_chunk.lstrip()

    if not os.path.exists(file_path):
        return 0, len(clean_chunk)

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
    except OSError:
        return 0, len(clean_chunk)

    # --- attempt 1: anchor on first 120 chars (more unique than 60) ----------
    anchor = stripped[:120]
    start = content.find(anchor)
    if start != -1:
        return start, start + len(clean_chunk)

    # --- attempt 2: normalise line-endings and retry -------------------------
    content_norm = content.replace("\r\n", "\n").replace("\r", "\n")
    anchor_norm = anchor.replace("\r\n", "\n").replace("\r", "\n")
    start = content_norm.find(anchor_norm)
    if start != -1:
        return start, start + len(clean_chunk)

    # --- attempt 3: collapse whitespace (handles minor re-formatting) --------
    _ws = re.compile(r"\s+")
    content_coll = _ws.sub(" ", content_norm)
    anchor_coll = _ws.sub(" ", anchor_norm)
    start = content_coll.find(anchor_coll)
    if start != -1:
        return start, start + len(clean_chunk)

    return 0, len(clean_chunk)


def _jaccard(a: str, b: str, ngram: int = 4) -> float:
    """Character n-gram Jaccard similarity — fast near-duplicate check."""
    sa = {a[i: i + ngram] for i in range(len(a) - ngram + 1)}
    sb = {b[i: i + ngram] for i in range(len(b) - ngram + 1)}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def deduplicate_chunks(
    chunks: list[str],
    metadata: list[dict[str, Any]],
    similarity_threshold: float = 0.85,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Drop near-duplicate chunks and keep the first representative."""
    kept_chunks: list[str] = []
    kept_meta: list[dict[str, Any]] = []

    for text, meta in zip(chunks, metadata):
        norm = re.sub(r"\s+", " ", _strip_injected_header(text)).strip()
        is_dup = any(
            _jaccard(norm, re.sub(r"\s+", " ", _strip_injected_header(k)).strip())
            >= similarity_threshold
            for k in kept_chunks
        )
        if not is_dup:
            kept_chunks.append(text)
            kept_meta.append(meta)

    return kept_chunks, kept_meta


def retrieve_chunks(
    query: str,
    found_files: list[MinimalSource],
    k: int = 5,
    bm25_save_path: str = "data/processed/bm25_index",
    overretrieve_factor: int = 3,
    dedup_threshold: float = 0.85,
) -> list[str]:
    """Retrieve the k most relevant chunks for query from the BM25 index."""
    if not os.path.exists(bm25_save_path):
        raise LLMException(
            f"BM25 index not found at '{bm25_save_path}'. Run indexing first."
        )

    retriever: Any = bm25s.BM25.load(bm25_save_path, load_corpus=True)  # type: ignore

    # query cleaning
    # Strip conversational fluff to focus BM25 heavily on actual technical variables
    clean_query = query.replace("in vLLM's", "").replace("in vLLM", "").replace("vLLM", "")
    clean_query = re.sub(r"^(What|How|Why|When|Where)( is| does|'s| are| happens)?\b", "", clean_query, flags=re.IGNORECASE)

    query_tokens: Any = bm25s.tokenize(clean_query, stopwords="en")  # type: ignore

    # over-retrieve so deduplication has candidates to thin out
    n_candidates = min(k * overretrieve_factor, 50)
    retrieval_output = retriever.retrieve(query_tokens, k=n_candidates)  # type: ignore

    raw_docs: list[dict[str, Any]] = (
        list(retrieval_output[0][0])
        if retrieval_output and len(retrieval_output[0]) > 0
        else []
    )

    if not raw_docs:
        return []

    raw_texts = [cast(str, d.get("text", "")) for d in raw_docs]
    raw_metas = [cast(dict[str, Any], d.get("metadata", {})) for d in raw_docs]

    # deduplication
    deduped_texts, deduped_metas = deduplicate_chunks(
        raw_texts, raw_metas, similarity_threshold=dedup_threshold
    )

    # take top-k after dedup
    final_texts = deduped_texts[:k]
    final_metas = deduped_metas[:k]

    # build output + populate found_files
    chunks: list[str] = []
    for text_val, meta_val in zip(final_texts, final_metas):
        raw_path = str(meta_val.get("source", "Unknown file"))
        clean_path = normalize_path(raw_path)
        real_path = get_real_path(clean_path)

        start_meta = meta_val.get("first_character_index")
        end_meta = meta_val.get("last_character_index")

        if start_meta is not None and end_meta is not None:
            start_idx, end_idx = int(start_meta), int(end_meta)
        else:
            start_idx, end_idx = locate_character_indices(real_path, text_val)

        found_files.append(
            MinimalSource(
                file_path=clean_path,
                first_character_index=start_idx,
                last_character_index=end_idx,
            )
        )
        chunks.append(text_val)

    return chunks


def index_files(codebase_dir: str, max_chunk_size: int = 2000) -> None:
    """Build and persist a BM25 index for the selected codebase."""
    log_message("Starting indexing...", LogType.INFO)
    from src.ingest import CodebaseIndexer  # local import to avoid circular deps

    indexer = CodebaseIndexer(
        codebase_dir=codebase_dir,
        max_chunk_size=max_chunk_size,
    )
    indexer.run_index()
    log_message("Indexing complete!", LogType.SUCCESS)