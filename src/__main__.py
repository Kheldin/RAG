import os
import json
import fire
import dspy
import chromadb
import bm25s
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Document, Metadata, ID, QueryResult, GetResult
from typing import Any, cast
from src.ingest import CodebaseIndexer


def setup_environment() -> None:
    """Configures the default LLM for DSPy."""
    ollama_qwen = dspy.LM(
        model="ollama/qwen3:0.6b", api_base="http://localhost:11434", api_key="none"
    )
    dspy.configure(lm=ollama_qwen)  # type: ignore


def load_retrievers(
    chroma_path: str, collection_name: str
) -> tuple[Collection, bm25s.BM25]:
    """Connects to ChromaDB and builds the in-memory BM25 index from its contents."""
    chroma_client: ClientAPI = chromadb.PersistentClient(path=chroma_path)
    collection: Collection = chroma_client.get_or_create_collection(
        name=collection_name
    )

    all_data: GetResult = collection.get()

    # Safely unwrap Optional types to satisfy Pylance
    all_docs: list[Document] = all_data.get("documents") or []
    all_metas: list[Metadata] = all_data.get("metadatas") or []
    all_ids: list[ID] = all_data.get("ids") or []

    print("Tokenizing and building bm25s index in memory...")
    corpus: list[dict[str, Any]] = [
        {"id": doc_id, "text": text, "metadata": meta}
        for doc_id, text, meta in zip(all_ids, all_docs, all_metas)
    ]

    corpus_tokens = bm25s.tokenize([doc["text"] for doc in corpus])  # type: ignore
    bm25_retriever = bm25s.BM25(corpus=corpus)
    bm25_retriever.index(corpus_tokens)  # type: ignore

    return collection, bm25_retriever


def hybrid_retrieve(
    question: str, k: int, collection: Collection, bm25_retriever: bm25s.BM25
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Core retrieval logic shared by both the RAG bot and the dataset searcher."""
    vector_results: QueryResult = collection.query(query_texts=[question], n_results=k)

    raw_docs = vector_results.get("documents")
    raw_metas = vector_results.get("metadatas")
    raw_ids = vector_results.get("ids")

    vec_docs: list[str] = raw_docs[0] if raw_docs and len(raw_docs) > 0 else []
    vec_metas: list[Metadata] = raw_metas[0] if raw_metas and len(raw_metas) > 0 else []
    vec_ids: list[str] = raw_ids[0] if raw_ids and len(raw_ids) > 0 else []

    query_tokens = bm25s.tokenize(question)  # type: ignore
    bm25_results, _ = bm25_retriever.retrieve(query_tokens, k=k)  # type: ignore

    combined_chunks: list[tuple[str, str]] = []
    seen_ids: set[str] = set()

    # Combine, rank, and deduplicate the results
    # The goal of deduplicating is to prevent
    #   feeding the exact same code chunk to the LLM twice
    #   when both the vector search (semantic)
    #   and BM25S (keyword) catch the same highly relevant file.

    # By filtering out duplicates, we:
    #   - Save context tokens
    #   - Reduce latency: Lower the time and computational cost it takes
    #       for the local LLM to process the prompt.
    #   - Improve accuracy: Prevent the LLM's attention mechanism
    #       from over-fixating on repetitive text.
    for doc_id, text, meta in zip(vec_ids, vec_docs, vec_metas):
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            source_file = (
                str(meta.get("source", "Unknown file")) if meta else "Unknown file"
            )
            combined_chunks.append((source_file, text))

    for match in bm25_results[0]:  # type: ignore
        doc_id = cast(str, match["id"])
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            meta = cast(dict[str, Any], match["metadata"])
            source_file = (
                str(meta.get("source", "Unknown file")) if meta else "Unknown file"
            )
            combined_chunks.append((source_file, cast(str, match["text"])))

    # Force the combined list to never exceed the requested 'k'
    combined_chunks = combined_chunks[:k]

    unique_sources: list[str] = list({source for source, _ in combined_chunks})
    context_texts: list[str] = [text for _, text in combined_chunks]

    return context_texts, unique_sources, combined_chunks


class CodebaseRAG(dspy.Module):
    """Hybrid Retrieval-Augmented Generation module for codebase querying."""

    def __init__(self, collection: Collection, bm25_retriever: bm25s.BM25) -> None:
        super().__init__()  # type: ignore
        self.collection = collection
        self.bm25_retriever = bm25_retriever

        self.generate_answer = dspy.ChainOfThought(
            "context, question -> answer",
            instructions="Answer the question using the provided codebase context. "
            "Explicitly mention the file names you used from the context headers.",  # type: ignore
        )

    def forward(self, question: str, k: int = 3) -> dspy.Prediction:
        """Executes the generation pipeline using the hybrid retriever."""
        context_texts, unique_sources, combined_chunks = hybrid_retrieve(
            question, k, self.collection, self.bm25_retriever
        )

        # Format context dynamically with file headers for the LLM
        formatted_context_list: list[str] = [
            f"--- File: {source} ---\n{text}\n" for source, text in combined_chunks
        ]
        context_str: str = "\n".join(formatted_context_list)

        prediction = self.generate_answer(context=context_str, question=question)  # type: ignore

        return dspy.Prediction(
            context=context_texts,
            sources=unique_sources,
            reasoning=getattr(prediction, "reasoning", ""),
            answer=getattr(prediction, "answer", ""),
        )


class CLICommands:
    """Exposes methods directly as command-line interfaces using Google Fire."""

    def answer(self, question: str, k: int = 10) -> None:
        """Answers a single query utilizing the hybrid retriever setup."""
        setup_environment()
        chroma_col, bm25_idx = load_retrievers(
            chroma_path="./my_local_chromadb", collection_name="codebase_chunks"
        )

        rag_bot = CodebaseRAG(collection=chroma_col, bm25_retriever=bm25_idx)
        result = rag_bot(question=question, k=k)

        print("\n=========================================")
        print(f"Question: {question}")
        print("=========================================")
        print(f"Thought Process:\n{getattr(result, 'reasoning', '')}\n")
        print(f"Final Answer:\n{getattr(result, 'answer', '')}\n")

        print("--- Sources Used ---")
        for idx, source in enumerate(getattr(result, "sources", []), 1):
            print(f"[{idx}] {source}")

    def index(
        self, codebase_dir: str = "vllm-0.10.1", max_chunk_size: int = 1000
    ) -> None:
        """Triggers scanning and text chunk extraction configurations."""
        indexer = CodebaseIndexer(
            codebase_dir=codebase_dir, max_chunk_size=max_chunk_size
        )
        indexer.run_index()

    def search_dataset(self, dataset_path: str, k: int = 10, save_directory: str = "data/output/search_results") -> None:
        """Reads a JSON dataset, searches for contexts for each question, and saves the results."""
        
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset not found at {dataset_path}")
            return

        with open(dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # Fix: Extract the inner list if the JSON is structured as a dictionary wrapper
        if isinstance(raw_data, dict):
            # Try finding the questions list using common keys, default to the dictionary values
            questions_list = raw_data.get("rag_questions") or raw_data.get("questions")
            if not isinstance(questions_list, list):
                print("Error: JSON is a dictionary but could not find a 'rag_questions' or 'questions' list inside.")
                return
        elif isinstance(raw_data, list):
            questions_list = raw_data
        else:
            print("Error: Unexpected JSON root structure. Expected a list or a dictionary wrapper.")
            return

        # Initialize Retrievers once for the whole dataset loop
        chroma_col, bm25_idx = load_retrievers(
            chroma_path="./my_local_chromadb", 
            collection_name="codebase_chunks"
        )

        processed_questions = []
        for item in questions_list:
            if not isinstance(item, dict):
                continue

            # Handle different common dataset keys for the query string
            question = item.get("question") or item.get("query")
            
            if not question:
                processed_questions.append(item)
                continue

            # Retrieve without hitting the LLM
            context_texts, unique_sources, _ = hybrid_retrieve(
                question=question, k=k, collection=chroma_col, bm25_retriever=bm25_idx
            )
            
            # Attach retrieved data back to the item object
            new_item = item.copy()
            new_item["retrieved_contexts"] = context_texts
            new_item["sources"] = unique_sources
            processed_questions.append(new_item)

        # Re-wrap into the original dictionary structure if it was a dictionary originally
        output_data = {"rag_questions": processed_questions} if isinstance(raw_data, dict) else processed_questions

        # Save to disk
        os.makedirs(save_directory, exist_ok=True)
        filename = os.path.basename(dataset_path)
        save_path = os.path.join(save_directory, filename)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)

        print(f"Saved student_search_results to {save_path}")


if __name__ == "__main__":
    fire.Fire(CLICommands)  # type: ignore
