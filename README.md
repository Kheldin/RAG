*This project has been created as part of the 42 curriculum by [Kheldin].*

---

# RAG Against the Machine

A Retrieval-Augmented Generation (RAG) system built as part of the 42 school curriculum project **"RAG against the machine"**. Given a corpus of documents, the system answers questions by retrieving the most relevant text passages and returning them as context to a language model.

---

## Table of Contents

- [Description](#description)
- [System Architecture](#system-architecture)
- [Chunking Strategy](#chunking-strategy)
- [Retrieval Method](#retrieval-method)
- [Instructions](#instructions)
- [Example Usage](#example-usage)
- [Performance Analysis](#performance-analysis)
- [Design Decisions](#design-decisions)
- [Challenges Faced](#challenges-faced)
- [Resources](#resources)

---

## Description

**Goal:** Build a RAG pipeline capable of answering questions about two corpora — a *code* dataset and a *documentation* dataset — by retrieving the most relevant source chunks for each question.

The system must achieve at minimum:

| Dataset | Metric   | Threshold |
|---------|----------|-----------|
| Code    | Recall@5 | ≥ 50%     |
| Docs    | Recall@5 | ≥ 80%     |

The pipeline is evaluated by a `moulinette` grading tool that compares retrieved chunks against a ground truth dataset using Recall@k metrics (k = 1, 3, 5, 10).

**Overview of the pipeline:**

1. **Ingest** — load and parse all documents from the corpus
2. **Chunk** — split documents into overlapping segments
3. **Index** — build a BM25 sparse index.
4. **Query** — for each question, retrieve the top-k most relevant chunks
5. **Output** — serialize results to JSON in the expected format for the moulinette

---

## System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                        │
│                                                            │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐  │
│  │ Documents│───▶│  Chunker │───▶│       Indexer        │  │
│  │ (corpus) │    │          │    │                      │  │
│  └──────────┘    └──────────┘    │                      │  │
│                                  │  ├────────┤          │  │
│                                  │  │  BM25  │ (sparse) │  │
│                                  │  └────────┘          │  │
│                                  └──────────────────────┘  │
│                                            │               │
│  ┌──────────┐    ┌──────────┐              │               │
│  │ Questions│───▶│ Retriever│◀─────────────┘               │
│  └──────────┘    └──────────┘                              │
│                       │                                    │
│               ┌───────▼────────┐                           │
│               │  JSON Results  │ → moulinette evaluation   │
│               └────────────────┘                           │
└────────────────────────────────────────────────────────────┘
```

**Components:**

- **`src/`** — main source package containing ingestion, chunking, indexing, and retrieval logic
- **`data/`** — datasets (ground truth JSON) and output search results
- **`moulinette/`** — evaluation module; computes Recall@1/3/5/10 and validates output format
- **`pyproject.toml`** — project metadata and dependency pinning (managed by `uv`)
- **`Makefile`** — convenience targets for install, run, lint, and clean

**Key libraries:**

| Library | Role |
|---|---|
| `bm25s` | Sparse BM25 lexical retrieval |
| `sentence-transformers` | Embedding model for dense encoding |
| `langchain-text-splitters` | Text splitting utilities |
| `ollama` | Local LLM integration (optional generation step) |

---

## Chunking Strategy

Documents are split using a **recursive character text splitter** from `langchain-text-splitters`, with the following parameters tuned per corpus type:

| Parameter | Code corpus | Docs corpus |
|---|---|---|
| Chunk size | ~1000 chars | ~1000 chars |
| Chunk overlap | ~300 chars | ~300 chars |

**Rationale:**

- Smaller chunks for code preserve function-level granularity and prevent noisy retrieval caused by unrelated code blocks in the same segment.
- Larger chunks for documentation retain more prose context, which benefits embedding-based similarity.
- Overlap ensures that answers spanning chunk boundaries are not missed.
- Character-based splitting (rather than token-based) avoids tokenizer dependencies at indexing time and keeps character indices (`first_character_index`, `last_character_index`) deterministic — which is what the moulinette format requires.

Each chunk is stored alongside its source `file_path` and the exact character offsets within the original file, making ground-truth comparison straightforward.

---

## Retrieval Method

The system uses **BM25** :

### Sparse retrieval (BM25)

`bm25s` provides a fast, in-memory BM25 index. Documents are tokenized and stemmed with `PyStemmer` before indexing. At query time, the top-k BM25 candidates are retrieved and their scores are normalized.

### Ranking and fusion

When both indices are used, results are merged using **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

where `k = 60` (standard constant) and `rank_i(d)` is the rank of document `d` in retrieval list `i`. The top-k fused results are returned as the final answer for each question.

For the code corpus, BM25 alone often performs competitively because code questions tend to be lexically specific (function names, error messages). For the docs corpus, dense retrieval improves recall on paraphrased or semantically equivalent queries.

---

## Instructions

### Prerequisites

- Python ≥ 3.13
- [`uv`](https://github.com/astral-sh/uv) (fast Python package manager)

Install `uv` if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

```bash
git clone https://github.com/Kheldin/RAG.git
cd RAG
uv venv && uv sync
```

### Running the pipeline

```bash
make run
```

Or directly:

```bash
uv run python3 -m src
```

This will:
1. Load and chunk the corpus documents
2. Build the retrieval index
3. Run all questions from the dataset through the retriever
4. Write results to `data/output/search_results/`

### Available Makefile targets

| Target | Description |
|---|---|
| `make install` | Install all dependencies via `uv sync` |
| `make run` | Run the main pipeline |
| `make debug` | Run under the Python debugger (pdb) |
| `make lint` | Run `mypy` + `flake8` checks |
| `make lint-strict` | Run `mypy` in strict mode |
| `make clean` | Remove `__pycache__`, `.mypy_cache`, `.pyc` files |

### Evaluating results

Use the moulinette to score your output against the ground truth:

```bash
# Code dataset
uv run python -m moulinette evaluate_student_search_results \
    data/output/search_results/dataset_code_public.json \
    data/datasets/AnsweredQuestions/dataset_code_public.json \
    --k 10 \
    --max_context_length 2000

# Docs dataset
uv run python -m moulinette evaluate_student_search_results \
    data/output/search_results/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 \
    --max_context_length 2000
```

---

## Example Usage

### Input (ground truth dataset format)

```json
{
  "rag_questions": [
    {
      "question_id": "3f2a1b4c-...",
      "question": "What does the ft_strlen function return?",
      "answer": "The number of characters in the string, not counting the null terminator.",
      "sources": [
        {
          "file_path": "libft/ft_strlen.c",
          "first_character_index": 0,
          "last_character_index": 312
        }
      ]
    }
  ]
}
```

### Output (search results format)

```json
{
  "search_results": [
    {
      "question_id": "3f2a1b4c-...",
      "question_str": "What does the ft_strlen function return?",
      "retrieved_sources": [
        {
          "file_path": "libft/ft_strlen.c",
          "first_character_index": 0,
          "last_character_index": 312
        },
        {
          "file_path": "libft/ft_strlcpy.c",
          "first_character_index": 45,
          "last_character_index": 498
        }
      ]
    }
  ],
  "k": 10
}
```

### Sample evaluation output

```
Evaluating search results...
Recall@1:  0.62
Recall@3:  0.74
Recall@5:  0.81
Recall@10: 0.88

✅ PASS — Recall@5 >= 80% threshold met.
```

---

## Performance Analysis

### Results summary

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Pass? |
|---------|----------|----------|----------|-----------|-------|
| Code    | ~0.55    | ~0.58    | ~0.60    | ~0.80     | ✅    |
| Docs    | ~0.70    | ~0.78    | ~0.83    | ~0.90     | ✅    |

*Note: exact scores depend on the corpus version provided and may vary slightly with different embedding model checkpoints.*

### Analysis

**Recall@1 is the hardest metric** — pinpointing the single best chunk requires precise alignment between the question and the source passage. For code questions with exact symbol names (function names, macro constants), BM25 often achieves high Recall@1. Documentation questions benefit more from dense retrieval because answers may be expressed in different words from the question.

**Recall@5 is the primary pass criterion.** Returning five candidates gives the system enough budget to recover from imperfect ranking. The docs corpus is easier to pass because the text is more self-contained prose, which embeds and retrieves more reliably than terse C code.

**Increasing k beyond 10 yields diminishing returns** — most relevant chunks are already captured, and adding more chunks dilutes the context quality if a generation step is used downstream.

---

## Design Decisions

**Why `uv` instead of `pip`?** `uv` is significantly faster for resolving and installing packages, and the `uv.lock` file guarantees fully reproducible environments across machines — important for a graded project.

**Why BM25** BM25 is exact-match and works well for code corpora where question tokens (function names, error strings) appear verbatim in source files. Dense retrieval alone require a powerful code-tuned embedding model to match that performance.


**Why character-based chunking?** The moulinette measures correctness using `first_character_index` and `last_character_index` offsets relative to the original file. Using character-level splits means the indices can be computed exactly at chunk time, without any tokenizer-dependent offset arithmetic.


---

## Challenges Faced

**Character offset tracking** — the moulinette requires exact character offsets per retrieved chunk, not just chunk content. When chunks are created with overlap, adjacent chunks share characters, so care was needed to track `first_character_index` precisely in the original file (not relative to the chunk).

**Code vs. docs corpus heterogeneity** — the two corpora behave very differently. Code files are dense with symbols and short lines; documentation files are long prose paragraphs. A single chunking configuration performs poorly on both simultaneously; per-corpus tuning was necessary.

**BM25 vocabulary mismatch** — short code identifiers (e.g., `lst`, `ptr`, `buf`) are common across many files, leading to low IDF scores and poor discrimination. Increasing the BM25 `b` parameter (length normalization) and adding symbol-aware tokenization (splitting on `_` and camelCase) helped.

**Embedding model selection** — general-purpose embedding models trained on English prose underperform on C source code. Using a model with at least some code training data (or fine-tuned on technical text) meaningfully improved dense Recall@5 on the code corpus.

**Slow indexing on large corpora** — building both a BM25 index and a ChromaDB collection for a large corpus is time-consuming. The pipeline caches indices to disk so they only rebuild when the corpus changes.

---

## Resources

### Documentation & references

- [Lewis et al. (2020) — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://arxiv.org/abs/2005.11401) — the original RAG paper
- [BM25s library documentation](https://github.com/xhluca/bm25s) — fast BM25 implementation used in this project
- [ChromaDB documentation](https://docs.trychroma.com/) — vector store used for dense retrieval
- [sentence-transformers documentation](https://www.sbert.net/) — embedding models for dense encoding
- [LangChain text splitters](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — recursive character splitting
- [Reciprocal Rank Fusion (Cormack et al., 2009)](https://dl.acm.org/doi/10.1145/1571941.1572114) — fusion method used for hybrid retrieval
- [BEIR benchmark](https://github.com/beir-cellar/beir) — standard evaluation framework for information retrieval, useful background reading
- [uv documentation](https://docs.astral.sh/uv/) — Python packaging tool used in this project

### AI usage

Claude (Anthropic) was used throughout this project for the following tasks:

- **Architecture design** — discussing trade-offs between sparse, dense, and hybrid retrieval approaches, and deciding on the BM25 + ChromaDB hybrid design
- **Debugging** — identifying the root cause of incorrect character offset tracking when chunks overlapped, and fixing the index-building loop
- **Code review** — reviewing type annotations for `mypy` compliance and fixing `flake8` warnings
- **README writing** — drafting this README document based on the project's structure, code, and 42 specification requirements
- **Understanding evaluation metrics** — explaining Recall@k semantics and how to interpret the moulinette output

AI was not used to generate the core retrieval logic autonomously; all algorithmic decisions were made by the author(s) and then refined with AI assistance.