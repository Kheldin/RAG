# eval.py
from typing import Any
import bm25s
from chromadb.api.models.Collection import Collection
from chromadb.api.types import QueryResult

from src.__main__ import load_retrievers


def evaluate_hybrid_retrieval(
    question: str, 
    collection: Collection, 
    bm25_retriever: bm25s.BM25, 
    k: int
) -> list[str]:
    """Runs the exact same hybrid retrieval loop as your RAG bot, bounded by k."""
    # Semantic Search
    vector_results: QueryResult = collection.query(query_texts=[question], n_results=k)
    raw_ids = vector_results.get("ids")
    vec_ids: list[str] = raw_ids[0] if raw_ids and len(raw_ids) > 0 else []

    # Keyword Search
    query_tokens = bm25s.tokenize(question)  # type: ignore
    bm25_results, _ = bm25_retriever.retrieve(query_tokens, k=k)

    # Combine and Deduplicate while preserving order rank
    retrieved_ids: list[str] = []
    seen_ids: set[str] = set()

    # Interleave or add sequentially (mimicking your forward loop)
    for doc_id in vec_ids:
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            retrieved_ids.append(doc_id)

    for match in bm25_results[0]:  # type: ignore
        doc_id = str(match["id"])
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            retrieved_ids.append(doc_id)

    # Return only up to the requested top_k boundary slice
    return retrieved_ids[:k]


def run_evaluation() -> None:
    # 1. Load the same indexed data
    chroma_col, bm25_idx = load_retrievers(
        chroma_path="./my_local_chromadb", 
        collection_name="codebase_chunks"
    )

    # 2. Get a sample corpus to construct a mock/real evaluation dataset
    # (In production, replace this with your curated golden evaluation dataset)
    all_data = chroma_col.get()
    all_ids: list[str] = all_data.get("ids") or []
    all_docs: list[str] = all_data.get("documents") or []
    
    if not all_ids:
        print("Error: The Chroma database is empty. Index some files first!")
        return

    print("\nPreparing evaluation dataset (100 test cases)...")
    eval_dataset: list[dict[str, Any]] = []
    
    # Generate 100 mock evaluation pairs from your actual data for demonstration
    for i in range(100):
        target_idx = i % len(all_ids)
        sample_text = all_docs[target_idx]
        
        # Take the first 5 words as a primitive query simulation
        mock_query = " ".join(sample_text.split()[:5]) or "query"
        
        eval_dataset.append({
            "question": f"Context inquiry: {mock_query}",
            "ground_truth_id": all_ids[target_idx]  # The specific chunk ID that holds the answer
        })

    # 3. Calculate Recall at different K boundaries
    k_values = [1, 3, 5, 10]
    recall_scores: dict[int, float] = {k: 0.0 for k in k_values}

    for item in eval_dataset:
        q = item["question"]
        gt = item["ground_truth_id"]

        for k in k_values:
            top_k_retrieved = evaluate_hybrid_retrieval(q, chroma_col, bm25_idx, k=k)
            # If the correct chunk ID is present anywhere within the top K, it's a hit
            if gt in top_k_retrieved:
                recall_scores[k] += 1.0

    # 4. Print results matching your output specifications exactly
    total_q = len(eval_dataset)
    print("\nEvaluation Results")
    print("========================================")
    print(f"Questions evaluated: {total_q}")
    for k in k_values:
        final_score = recall_scores[k] / total_q
        print(f"Recall@{k}: {final_score:.3f}")


if __name__ == "__main__":
    run_evaluation()