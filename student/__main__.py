import fire
import dspy
import chromadb
import bm25s
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Document, Metadata, ID, QueryResult, GetResult
from typing import Any, cast


def setup_environment() -> None:
    """Configures the default LLM for DSPy."""
    ollama_qwen = dspy.LM(
        model="ollama/qwen3:0.6b", 
        api_base="http://localhost:11434", 
        api_key="none"
    )
    dspy.configure(lm=ollama_qwen)  # type: ignore


def load_retrievers(chroma_path: str, collection_name: str) -> tuple[Collection, bm25s.BM25]:
    """Connects to ChromaDB and builds the in-memory BM25 index from its contents."""
    chroma_client: ClientAPI = chromadb.PersistentClient(path=chroma_path)
    collection: Collection = chroma_client.get_or_create_collection(name=collection_name)

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


class CodebaseRAG(dspy.Module):
    """Hybrid Retrieval-Augmented Generation module for codebase querying."""
    
    def __init__(self, collection: Collection, bm25_retriever: bm25s.BM25) -> None:
        super().__init__()  # type: ignore
        self.collection = collection
        self.bm25_retriever = bm25_retriever

        self.generate_answer = dspy.ChainOfThought(
            "context, question -> answer",
            instructions="Answer the question using the provided codebase context. "
                         "Explicitly mention the file names you used from the context headers."  # type: ignore
        )

    def forward(self, question: str, k: int = 3) -> dspy.Prediction:
        """Executes the hybrid RAG retrieval and generation pipeline for a given query."""
        
        # Query ChromaDB natively for Vector/Semantic Search
        vector_results: QueryResult = self.collection.query(
            query_texts=[question], 
            n_results=k
        )
        
        raw_docs = vector_results.get("documents")
        raw_metas = vector_results.get("metadatas")
        raw_ids = vector_results.get("ids")
        
        # Safely unwrap the nested lists without unnecessary casts
        vec_docs: list[str] = raw_docs[0] if raw_docs and len(raw_docs) > 0 else []
        vec_metas: list[Metadata] = raw_metas[0] if raw_metas and len(raw_metas) > 0 else []
        vec_ids: list[str] = raw_ids[0] if raw_ids and len(raw_ids) > 0 else []

        # Query BM25S natively for Exact Keyword Search
        query_tokens = bm25s.tokenize(question)  # type: ignore
        bm25_results, _ = self.bm25_retriever.retrieve(query_tokens, k=k)
        
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
        
        combined_chunks: list[tuple[str, str]] = []
        seen_ids: set[str] = set()

        # Add vector results first
        for doc_id, text, meta in zip(vec_ids, vec_docs, vec_metas):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                source_file = str(meta.get("source", "Unknown file")) if meta else "Unknown file"
                combined_chunks.append((source_file, text))

        # Add BM25S results
        for match in bm25_results[0]:  # type: ignore
            doc_id = cast(str, match["id"])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                meta = cast(dict[str, Any], match["metadata"])
                source_file = str(meta.get("source", "Unknown file")) if meta else "Unknown file"
                combined_chunks.append((source_file, cast(str, match["text"])))

        combined_chunks = combined_chunks[:k+1]
        
        unique_sources: set[str] = {source for source, _ in combined_chunks}

        # Format context dynamically with file headers
        formatted_context_list: list[str] = [
            f"--- File: {source} ---\n{text}\n" 
            for source, text in combined_chunks
        ]
        context_str: str = "\n".join(formatted_context_list)
        context_texts: list[str] = [text for _, text in combined_chunks]

        # Step E: Pass everything to the LLM
        prediction = self.generate_answer(context=context_str, question=question)  # type: ignore

        return dspy.Prediction(
            context=context_texts,
            sources=list(unique_sources),
            reasoning=getattr(prediction, "reasoning", ""),
            answer=getattr(prediction, "answer", ""),
        )


class CLICommands:
    """Exposes methods directly as command-line interfaces using Google Fire."""

    def answer(self, question: str, k: int = 10) -> None:
        """Answers a single query utilizing the hybrid retriever setup."""
        setup_environment()
        chroma_col, bm25_idx = load_retrievers(
            chroma_path="./my_local_chromadb", 
            collection_name="codebase_chunks"
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


if __name__ == "__main__":
    fire.Fire(CLICommands)