import os
import sys
import json
import subprocess
import time
import socket
from typing import Any, cast

import fire  # type: ignore
import dspy  # type: ignore
import chromadb
import bm25s  # type: ignore
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection  # type: ignore
from chromadb.api.types import Document, Metadata, ID, QueryResult, GetResult

from src.models.models import (
    MinimalAnswer, 
    MinimalSearchResults, 
    MinimalSource, 
    AnsweredQuestion, 
    StudentSearchResultsAndAnswer, 
    StudentSearchResults
)

def is_ollama_alive(host: str = "127.0.0.1", port: int = 11434) -> bool:
    """Checks if something is listening on the local Ollama port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def setup_environment() -> None:
    """Launches Ollama via background subprocess if not running, then configures DSPy."""
    model_name = "qwen3:0.6b"

    if not is_ollama_alive():
        print("Ollama server is not running. Launching background subprocess...")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid if os.name != "nt" else None,
            )

            print("Waiting for Ollama to wake up...")
            attempts = 0
            while not is_ollama_alive():
                time.sleep(1)
                attempts += 1
                if attempts > 15:
                    print("Error: Ollama took too long to respond. Ensure it's installed.")
                    sys.exit(1)
            print("Ollama server successfully launched!")
        except FileNotFoundError:
            print("Error: The 'ollama' executable was not found in your system PATH.")
            sys.exit(1)

    print(f"Ensuring model '{model_name}' is loaded...")
    try:
        subprocess.run(
            ["ollama", "pull", model_name], check=True, stdout=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        print(f"Warning: Failed to execute 'ollama pull {model_name}'. Proceeding anyway...")

    ollama_qwen: Any = dspy.LM(
        model=f"ollama/{model_name}", api_base="http://localhost:11434", api_key="none"
    )
    dspy.configure(lm=ollama_qwen)  # type: ignore


def normalize_path(path: str) -> str:
    """Strips any root codebase directory prefixes so paths conform to validation formats."""
    path = path.replace("\\", "/")
    prefixes = ["vllm-0.10.1/", "./vllm-0.10.1/"]
    for prefix in prefixes:
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def get_real_path(normalized_path: str, codebase_dir: str = "vllm-0.10.1") -> str:
    """Ensures a valid disk-location target path for locating text chunk string segments."""
    if os.path.exists(normalized_path):
        return normalized_path
    joined = os.path.join(codebase_dir, normalized_path)
    if os.path.exists(joined):
        return joined
    return normalized_path


def load_retrievers(
    chroma_path: str = "data/processed/chunks", 
    collection_name: str = "codebase_chunks", 
    bm25_save_path: str = "data/processed/bm25_index"
) -> tuple[Collection, Any]:
    """Connects to ChromaDB and loads BM25 from disk, or builds it if missing."""
    os.makedirs(chroma_path, exist_ok=True)
    chroma_client: ClientAPI = chromadb.PersistentClient(path=chroma_path)
    collection: Collection = chroma_client.get_or_create_collection(name=collection_name)

    if os.path.exists(bm25_save_path):
        bm25_retriever: Any = bm25s.BM25.load(bm25_save_path, load_corpus=True)  # type: ignore
        return collection, bm25_retriever

    print("BM25 index not found on disk. Building from ChromaDB (this will be slow)...")
    all_data: GetResult = collection.get()
    
    all_docs: list[Document] = cast(list[Document], all_data.get("documents") or [])
    all_metas: list[Metadata] = cast(list[Metadata], all_data.get("metadatas") or [])
    all_ids: list[ID] = cast(list[ID], all_data.get("ids") or [])

    if not all_docs:
        print("WARNING: ChromaDB is empty! BM25 will have no corpus.")
        corpus = [{"id": "dummy", "text": "dummy", "metadata": {}}]
    else:
        corpus = [
            {"id": doc_id, "text": text, "metadata": meta}
            for doc_id, text, meta in zip(all_ids, all_docs, all_metas)
        ]

    corpus_tokens: Any = bm25s.tokenize([doc["text"] for doc in corpus])  # type: ignore
    bm25_retriever = bm25s.BM25(corpus=corpus)
    bm25_retriever.index(corpus_tokens)  # type: ignore

    os.makedirs(os.path.dirname(bm25_save_path), exist_ok=True)
    bm25_retriever.save(bm25_save_path, corpus=corpus)  # type: ignore
    return collection, bm25_retriever


def locate_character_indices(file_path: str, chunk_text: str) -> tuple[int, int]:
    """Reads the source file to find the exact character indexes of the chunk text."""
    if not os.path.exists(file_path):
        return 0, len(chunk_text)
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        start_idx = content.find(chunk_text)
        if start_idx == -1:
            # Fallback handling windows carriage-return string alignments
            content_norm = content.replace("\r", "")
            chunk_norm = chunk_text.replace("\r", "")
            start_idx = content_norm.find(chunk_norm)
            if start_idx == -1:
                return 0, len(chunk_text)
        return start_idx, start_idx + len(chunk_text)
    except Exception:
        return 0, len(chunk_text)


def hybrid_retrieve(
    question: str, k: int, collection: Collection, bm25_retriever: Any
) -> tuple[list[str], list[MinimalSource], list[tuple[str, str]]]:
    """Core retrieval logic that dynamically builds typed MinimalSource items."""
    vector_results: QueryResult = collection.query(query_texts=[question], n_results=k)

    raw_docs = vector_results.get("documents")
    raw_metas = vector_results.get("metadatas")
    raw_ids = vector_results.get("ids")

    vec_docs: list[str] = cast(list[str], raw_docs[0]) if raw_docs is not None and len(raw_docs) > 0 else []
    vec_metas: list[dict[str, Any]] = cast(list[dict[str, Any]], raw_metas[0]) if raw_metas is not None and len(raw_metas) > 0 else []
    vec_ids: list[str] = cast(list[str], raw_ids[0]) if raw_ids is not None and len(raw_ids) > 0 else []

    query_tokens: Any = bm25s.tokenize(question)  # type: ignore
    retrieval_output: Any = bm25_retriever.retrieve(query_tokens, k=k)  # type: ignore
    bm25_results: list[dict[str, Any]] = cast(list[dict[str, Any]], retrieval_output[0][0]) if retrieval_output else []

    combined_raw_data: list[tuple[str, str, dict[str, Any]]] = []
    seen_ids: set[str] = set()

    for doc_id, text, meta in zip(vec_ids, vec_docs, vec_metas):
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            meta_dict = cast(dict[str, Any], meta) if meta else {}
            combined_raw_data.append((doc_id, text, meta_dict))

    for match in bm25_results:
        doc_id_match = cast(str, match.get("id", ""))
        if doc_id_match and doc_id_match not in seen_ids:
            seen_ids.add(doc_id_match)
            combined_raw_data.append(
                (
                    doc_id_match,
                    cast(str, match.get("text", "")),
                    cast(dict[str, Any], match.get("metadata", {})),
                )
            )

    combined_raw_data = combined_raw_data[:k]

    context_texts: list[str] = []
    minimal_sources: list[MinimalSource] = []
    rag_context_tuples: list[tuple[str, str]] = []

    for _, text_val, meta_val in combined_raw_data:
        raw_path = str(meta_val.get("source", "Unknown file"))
        clean_path = normalize_path(raw_path)
        real_path = get_real_path(clean_path)

        start_meta = meta_val.get("first_character_index")
        end_meta = meta_val.get("last_character_index")

        if start_meta is not None and end_meta is not None:
            start_idx, end_idx = int(start_meta), int(end_meta)
        else:
            start_idx, end_idx = locate_character_indices(real_path, text_val)

        context_texts.append(text_val)
        rag_context_tuples.append((clean_path, text_val))
        minimal_sources.append(
            MinimalSource(
                file_path=clean_path,
                first_character_index=start_idx,
                last_character_index=end_idx,
            )
        )

    return context_texts, minimal_sources, rag_context_tuples


class CodebaseRAG(dspy.Module):  # type: ignore
    """Hybrid Retrieval-Augmented Generation module for codebase querying."""

    def __init__(self, collection: Collection, bm25_retriever: Any) -> None:
        super().__init__()
        self.collection: Collection = collection
        self.bm25_retriever: Any = bm25_retriever

        self.generate_answer: Any = dspy.ChainOfThought(
            "context, question -> answer",
            instructions="Answer the question using the provided codebase context. Explicitly mention the file names you used from the context headers.",
        )

    def forward(self, question: str, k: int = 3) -> Any:
        context_texts, minimal_sources, combined_chunks = hybrid_retrieve(
            question, k, self.collection, self.bm25_retriever
        )

        formatted_context_list: list[str] = [
            f"--- File: {source} ---\n{text}\n" for source, text in combined_chunks
        ]
        context_str: str = "\n".join(formatted_context_list)

        prediction: Any = self.generate_answer(context=context_str, question=question)

        return dspy.Prediction(
            context=context_texts,
            sources=minimal_sources,
            reasoning=str(getattr(prediction, "reasoning", "")),
            answer=str(getattr(prediction, "answer", "")),
        )


class CLICommands:
    """Exposes methods directly as command-line interfaces using Google Fire."""

    def answer(
        self, 
        question: str, 
        k: int = 3, 
        save_directory: str = "data/output/search_results_and_answer"
    ) -> None:
        setup_environment()
        chroma_col, bm25_idx = load_retrievers()

        rag_bot = CodebaseRAG(collection=chroma_col, bm25_retriever=bm25_idx)
        result: Any = rag_bot(question=question, k=k)

        answer_text = str(getattr(result, "answer", ""))

        answer_res = MinimalAnswer(
            question_id="single_query",
            question_str=question,
            retrieved_sources=cast(list[MinimalSource], getattr(result, "sources", [])),
            answer=answer_text,
        )

        output_payload = StudentSearchResultsAndAnswer(
            search_results=[answer_res], 
            k=k
        )

        print("\n" + "=" * 40)
        print("ANSWER:")
        print("=" * 40)
        print(answer_text)
        print("=" * 40 + "\n")

        os.makedirs(save_directory, exist_ok=True)
        timestamp = int(time.time())
        save_path = os.path.join(save_directory, f"single_answer_{timestamp}.json")
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(output_payload.model_dump_json(indent=4))
            
        print(f"Saved payload structure to {save_path}")

    def index(
        self, codebase_dir: str = "data/raw/vllm-0.10.1", max_chunk_size: int = 1000
    ) -> None:
        from src.ingest import CodebaseIndexer  # type: ignore
        indexer: Any = CodebaseIndexer(
            codebase_dir=codebase_dir, max_chunk_size=max_chunk_size
        )
        indexer.run_index()

        print("Pre-building and saving BM25 index to disk...")
        _, _ = load_retrievers()

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset not found at {dataset_path}")
            return

        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_data: Any = json.load(f)

        questions_list: list[dict[str, Any]] = []
        if isinstance(raw_data, dict):
            questions_list = cast(
                list[dict[str, Any]],
                raw_data.get("rag_questions") or raw_data.get("questions") or [],
            )
        elif isinstance(raw_data, list):
            questions_list = cast(list[dict[str, Any]], raw_data)
            
        if not questions_list:
            print(f"Error: No questions found inside the JSON at {dataset_path}")
            return

        try:
            chroma_col, bm25_idx = load_retrievers()
        except Exception as e:
            print(f"CRITICAL ERROR loading databases: {e}")
            return

        search_results_list: list[MinimalSearchResults] = []

        for item in questions_list:
            if not isinstance(item, dict):
                continue
            q_text = str(item.get("question") or item.get("query", ""))
            q_id = str(item.get("question_id") or item.get("id", "unknown"))
            if not q_text:
                continue

            try:
                _, minimal_sources, _ = hybrid_retrieve(
                    question=q_text, k=k, collection=chroma_col, bm25_retriever=bm25_idx
                )
            except Exception as e:
                print(f"Retrieval error on question {q_id}: {e}")
                minimal_sources = []
                
            search_results_list.append(
                MinimalSearchResults(
                    question_id=q_id, question_str=q_text, retrieved_sources=minimal_sources
                )
            )

        final_output_model = StudentSearchResults(
            search_results=search_results_list, k=k
        )
        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, os.path.basename(dataset_path))

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(final_output_model.model_dump_json(indent=4))
        print(f"Saved student_search_results to {save_path}")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        if not os.path.exists(student_search_results_path):
            print(f"Error: Search results file not found at {student_search_results_path}")
            return

        with open(student_search_results_path, "r", encoding="utf-8") as f:
            raw_data: Any = json.load(f)

        try:
            search_data = StudentSearchResults.model_validate(raw_data)
        except Exception as e:
            print(f"Error parsing JSON against Pydantic schema: {e}")
            return

        questions_list = search_data.search_results
        k = search_data.k
        total_q = len(questions_list)

        print(f"Loaded {total_q} questions from {student_search_results_path}")

        setup_environment()

        generator: Any = dspy.ChainOfThought(
            "context, question -> answer",
            instructions="Answer the question using the provided codebase context. Explicitly mention the file names you used from the context headers.",
        )

        minimal_answers_list: list[MinimalAnswer] = []
        file_content_cache: dict[str, str] = {}

        for idx, item in enumerate(questions_list, 1):
            context_chunks: list[str] = []
            for src in item.retrieved_sources:
                if src.file_path not in file_content_cache:
                    try:
                        real_p = get_real_path(src.file_path)
                        with open(real_p, "r", encoding="utf-8", errors="ignore") as f:
                            file_content_cache[src.file_path] = f.read()
                    except Exception:
                        file_content_cache[src.file_path] = ""

                content = file_content_cache[src.file_path]
                if content:
                    chunk_text = content[src.first_character_index : src.last_character_index]
                    context_chunks.append(f"--- File: {src.file_path} ---\n{chunk_text}\n")

            context_str = "\n".join(context_chunks)
            prediction: Any = generator(context=context_str, question=item.question_str)
            answer_text = str(getattr(prediction, "answer", ""))

            minimal_answers_list.append(
                MinimalAnswer(
                    question_id=item.question_id,
                    question_str=item.question_str,
                    retrieved_sources=item.retrieved_sources,
                    answer=answer_text,
                )
            )

            sys.stdout.write(f"\rProcessed {idx} of {total_q} questions")
            sys.stdout.flush()

        print()

        final_output_model = StudentSearchResultsAndAnswer(
            search_results=minimal_answers_list,
            k=k,
        )

        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, os.path.basename(student_search_results_path))

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(final_output_model.model_dump_json(indent=4))

        print(f"Saved student_search_results_and_answer to {save_path}")


if __name__ == "__main__":
    fire.Fire(CLICommands)  # type: ignore