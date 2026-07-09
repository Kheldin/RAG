import os
import re
import bm25s # type: ignore
from typing import Any, cast
from src.models.models import MinimalSource
from .exceptions import LLMException, LogType, log_message

def normalize_path(path: str) -> str:
    path = path.replace("\\", "/")
    prefixes = ["data/raw/vllm-0.10.1/", "vllm-0.10.1/", "./vllm-0.10.1/"]
    for prefix in prefixes:
        if path.startswith(prefix):
            return path[len(prefix):]
    return path

def get_real_path(normalized_path: str) -> str:
    """Ensures we can find the physical file on disk to calculate character indices."""
    possible_bases = [
        "",
        "data/raw/vllm-0.10.1/",
        "vllm-0.10.1/"
    ]
    for base in possible_bases:
        joined = os.path.join(base, normalized_path)
        if os.path.exists(joined):
            return joined
    return normalized_path

def locate_character_indices(file_path: str, chunk_text: str) -> tuple[int, int]:
    if not os.path.exists(file_path):
        return 0, len(chunk_text)
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        # Strip our injected header
        clean_chunk = re.sub(r"^--- File: .*? ---\n", "", chunk_text)
        
        # 🚨 THE FIX: LangChain modifies trailing whitespace. 
        # We only use the first 60 characters to lock in the exact start index!
        snippet = clean_chunk.lstrip()[:60]
        
        start_idx = content.find(snippet)
        if start_idx == -1:
            start_idx = content.replace("\r", "").find(snippet.replace("\r", ""))
            if start_idx == -1:
                return 0, len(clean_chunk)
                
        return start_idx, start_idx + len(clean_chunk)
    except Exception:
        return 0, len(chunk_text)

def retrieve_chunks(
    query: str, 
    found_files: list[MinimalSource], 
    k: int = 3, 
    bm25_save_path: str = "data/processed/bm25_index"
) -> list[str]:
    """Retrieves chunks and modifies found_files in-place as expected by AI class."""
    if not os.path.exists(bm25_save_path):
        raise LLMException(f"BM25 index not found at {bm25_save_path}. Run index first.")
    
    retriever: Any = bm25s.BM25.load(bm25_save_path, load_corpus=True) # type: ignore

    query_tokens: Any = bm25s.tokenize(query) # type: ignore
    retrieval_output = retriever.retrieve(query_tokens, k=k) # type: ignore
    
    retrieved_docs: list[dict[str, Any]] = list(retrieval_output[0][0]) if retrieval_output and len(retrieval_output[0]) > 0 else []
    
    chunks = []
    for match in retrieved_docs:
        text_val = cast(str, match.get("text", ""))
        meta_val = cast(dict[str, Any], match.get("metadata", {}))

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
    """Stub calling your original indexer logic."""
    log_message("Starting indexing...", LogType.INFO)
    from src.ingest import CodebaseIndexer 
    indexer = CodebaseIndexer(codebase_dir=codebase_dir, max_chunk_size=max_chunk_size)
    indexer.run_index()
    log_message("Indexing complete!", LogType.SUCCESS)