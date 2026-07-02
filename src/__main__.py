import os
import sys
import json
import fire
import dspy
import chromadb
import bm25s
import subprocess
import time
import socket
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Document, Metadata, ID, QueryResult, GetResult
from typing import Any, cast

from src.models.models import (
    MinimalSource,
    MinimalSearchResults,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.ingest import CodebaseIndexer


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
                    print(
                        "Error: Ollama took too long to respond. Ensure it's installed and runnable via CLI."
                    )
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
        print(
            f"Warning: Failed to execute 'ollama pull {model_name}'. Proceeding anyway..."
        )

    ollama_qwen = dspy.LM(
        model=f"ollama/{model_name}", api_base="http://localhost:11434", api_key="none"
    )
    dspy.configure(lm=ollama_qwen)  # type: ignore


def load_retrievers(
    chroma_path: str, collection_name: str, bm25_save_path: str = "./my_local_bm25"
) -> tuple[Collection, bm25s.BM25]:
    """Connects to ChromaDB and loads BM25 from disk, or builds it if missing."""
    chroma_client: ClientAPI = chromadb.PersistentClient(path=chroma_path)
    collection: Collection = chroma_client.get_or_create_collection(
        name=collection_name
    )

    if os.path.exists(bm25_save_path):
        bm25_retriever = bm25s.BM25.load(bm25_save_path, load_corpus=True)  # type: ignore
        return collection, bm25_retriever

    print("BM25 index not found on disk. Building from ChromaDB (this will be slow)...")
    all_data: GetResult = collection.get()
    all_docs: list[Document] = all_data.get("documents") or []
    all_metas: list[Metadata] = all_data.get("metadatas") or []
    all_ids: list[ID] = all_data.get("ids") or []

    corpus: list[dict[str, Any]] = [
        {"id": doc_id, "text": text, "metadata": meta}
        for doc_id, text, meta in zip(all_ids, all_docs, all_metas)
    ]

    corpus_tokens = bm25s.tokenize([doc["text"] for doc in corpus])  # type: ignore
    bm25_retriever = bm25s.BM25(corpus=corpus)
    bm25_retriever.index(corpus_tokens)  # type: ignore

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
            return 0, len(chunk_text)
        return start_idx, start_idx + len(chunk_text)
    except Exception:
        return 0, len(chunk_text)


def hybrid_retrieve(
    question: str, k: int, collection: Collection, bm25_retriever: bm25s.BM25
) -> tuple[list[str], list[MinimalSource], list[tuple[str, str]]]:
    """Core retrieval logic that dynamically builds typed MinimalSource items."""
    vector_results: QueryResult = collection.query(query_texts=[question], n_results=k)

    raw_docs = vector_results.get("documents")
    raw_metas = vector_results.get("metadatas")
    raw_ids = vector_results.get("ids")

    vec_docs: list[str] = raw_docs[0] if raw_docs and len(raw_docs) > 0 else []
    vec_metas: list[Metadata] = raw_metas[0] if raw_metas and len(raw_metas) > 0 else []
    vec_ids: list[str] = raw_ids[0] if raw_ids and len(raw_ids) > 0 else []

    query_tokens = bm25s.tokenize(question)  # type: ignore
    bm25_results, _ = bm25_retriever.retrieve(query_tokens, k=k)  # type: ignore

    combined_raw_data: list[tuple[str, str, dict[str, Any]]] = []
    seen_ids: set[str] = set()

    for doc_id, text, meta in zip(vec_ids, vec_docs, vec_metas):
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            meta_dict = cast(dict[str, Any], meta) if meta else {}
            combined_raw_data.append((doc_id, text, meta_dict))

    for match in bm25_results[0]:  # type: ignore
        doc_id = cast(str, match["id"])
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            combined_raw_data.append(
                (
                    doc_id,
                    cast(str, match["text"]),
                    cast(dict[str, Any], match["metadata"]),
                )
            )

    combined_raw_data = combined_raw_data[:k]

    context_texts: list[str] = []
    minimal_sources: list[MinimalSource] = []
    rag_context_tuples: list[tuple[str, str]] = []

    for _, text, meta in combined_raw_data:
        file_path = str(meta.get("source", "Unknown file"))
        start_idx, end_idx = locate_character_indices(file_path, text)

        context_texts.append(text)
        rag_context_tuples.append((file_path, text))
        minimal_sources.append(
            MinimalSource(
                file_path=file_path,
                first_character_index=start_idx,
                last_character_index=end_idx,
            )
        )

    return context_texts, minimal_sources, rag_context_tuples


class CodebaseRAG(dspy.Module):
    """Hybrid Retrieval-Augmented Generation module for codebase querying."""

    def __init__(self, collection: Collection, bm25_retriever: bm25s.BM25) -> None:
        super().__init__()  # type: ignore
        self.collection = collection
        self.bm25_retriever = bm25_retriever

        self.generate_answer = dspy.ChainOfThought(
            "context, question -> answer",
            instructions="Answer the question using the provided codebase context. Explicitly mention the file names you used from the context headers.",  # type: ignore
        )

    def forward(self, question: str, k: int = 3) -> dspy.Prediction:
        context_texts, minimal_sources, combined_chunks = hybrid_retrieve(
            question, k, self.collection, self.bm25_retriever
        )

        formatted_context_list: list[str] = [
            f"--- File: {source} ---\n{text}\n" for source, text in combined_chunks
        ]
        context_str: str = "\n".join(formatted_context_list)

        prediction = self.generate_answer(context=context_str, question=question)  # type: ignore

        return dspy.Prediction(
            context=context_texts,
            sources=minimal_sources,
            reasoning=getattr(prediction, "reasoning", ""),
            answer=getattr(prediction, "answer", ""),
        )


class CLICommands:
    """Exposes methods directly as command-line interfaces using Google Fire."""

    def answer(self, question: str, k: int = 3) -> None:
        setup_environment()
        chroma_col, bm25_idx = load_retrievers(
            chroma_path="./my_local_chromadb", collection_name="codebase_chunks"
        )

        rag_bot = CodebaseRAG(collection=chroma_col, bm25_retriever=bm25_idx)
        result = rag_bot(question=question, k=k)

        search_res = MinimalSearchResults(
            question_id="single_query",
            question=question,
            retrieved_sources=getattr(result, "sources", []),
        )
        answer_res = MinimalAnswer(
            question_id="single_query",
            question=question,
            retrieved_sources=getattr(result, "sources", []),
            answer=getattr(result, "answer", ""),
        )

        output_payload = StudentSearchResultsAndAnswer(
            search_results=[search_res], search_results_and_answer=[answer_res], k=k
        )
        print(output_payload.model_dump_json(indent=4))

    def index(
        self, codebase_dir: str = "vllm-0.10.1", max_chunk_size: int = 1000
    ) -> None:
        indexer = CodebaseIndexer(
            codebase_dir=codebase_dir, max_chunk_size=max_chunk_size
        )
        indexer.run_index()

        print("Pre-building and saving BM25 index to disk...")
        _, _ = load_retrievers(
            chroma_path="./my_local_chromadb",
            collection_name="codebase_chunks",
            bm25_save_path="./my_local_bm25",
        )

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
            raw_data = json.load(f)

        questions_list: list[dict[str, Any]] = []
        if isinstance(raw_data, dict):
            questions_list = cast(
                list[dict[str, Any]],
                raw_data.get("rag_questions") or raw_data.get("questions") or [],
            )
        elif isinstance(raw_data, list):
            questions_list = cast(list[dict[str, Any]], raw_data)
        else:
            return

        if not questions_list:
            return

        chroma_col, bm25_idx = load_retrievers(
            chroma_path="./my_local_chromadb", collection_name="codebase_chunks"
        )
        search_results_list: list[MinimalSearchResults] = []

        for item in questions_list:
            if not isinstance(item, dict):
                continue
            q_text = str(item.get("question") or item.get("query", ""))
            q_id = str(item.get("question_id") or item.get("id", "unknown"))
            if not q_text:
                continue

            _, minimal_sources, _ = hybrid_retrieve(
                question=q_text, k=k, collection=chroma_col, bm25_retriever=bm25_idx
            )
            search_results_list.append(
                MinimalSearchResults(
                    question_id=q_id, question=q_text, retrieved_sources=minimal_sources
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
            print(
                f"Error: Search results file not found at {student_search_results_path}"
            )
            return

        with open(student_search_results_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

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

        generator = dspy.ChainOfThought(
            "context, question -> answer",
            instructions="Answer the question using the provided codebase context. Explicitly mention the file names you used from the context headers.",  # type: ignore
        )

        minimal_answers_list: list[MinimalAnswer] = []
        file_content_cache: dict[str, str] = {}

        for idx, item in enumerate(questions_list, 1):
            context_chunks = []
            for src in item.retrieved_sources:
                if src.file_path not in file_content_cache:
                    try:
                        with open(
                            src.file_path, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            file_content_cache[src.file_path] = f.read()
                    except Exception:
                        file_content_cache[src.file_path] = ""

                content = file_content_cache[src.file_path]
                if content:
                    chunk_text = content[
                        src.first_character_index : src.last_character_index
                    ]
                    context_chunks.append(
                        f"--- File: {src.file_path} ---\n{chunk_text}\n"
                    )

            context_str = "\n".join(context_chunks)
            prediction = generator(context=context_str, question=item.question)  # type: ignore
            answer_text = str(getattr(prediction, "answer", ""))

            minimal_answers_list.append(
                MinimalAnswer(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=item.retrieved_sources,
                    answer=answer_text,
                )
            )

            sys.stdout.write(f"\rProcessed {idx} of {total_q} questions")
            sys.stdout.flush()

        print()

        final_output_model = StudentSearchResultsAndAnswer(
            search_results=cast(list[MinimalSearchResults], minimal_answers_list),
            search_results_and_answer=minimal_answers_list,
            k=k,
        )

        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(
            save_directory, os.path.basename(student_search_results_path)
        )

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(final_output_model.model_dump_json(indent=4))

        print(f"Saved student_search_results_and_answer to {save_path}")


if __name__ == "__main__":
    fire.Fire(CLICommands)  # type: ignore
