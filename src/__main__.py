#!/usr/bin/env python3
"""Command-line entry point for indexing, retrieval, and answer generation."""

import inspect
import sys
import json
import os
import uuid
import fire # type: ignore
from collections.abc import Callable
from typing import Any
from tqdm import tqdm

from src.recall_metrics import evaluate, evaluate_student_search_results
from src.indexer import index_files, retrieve_chunks
from src.exceptions import LLMException, LogType, log_message
from src.models.models import (
    MinimalSource,
    StudentSearchResults,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResultsAndAnswer,
    MinimalAnswer,
)
from src.ai import AI

CommandMap = dict[str, Callable[..., Any]]

def validate_args(cli_map: CommandMap) -> None:
    """Reject unknown CLI flags before dispatching to Fire."""
    argv = sys.argv[1:]
    if not argv:
        return

    command = argv[0]
    if command not in cli_map:
        return
    func = cli_map[command]

    sig = inspect.signature(func)
    valid_params = set(sig.parameters.keys())

    passed_flags = {
        arg.split("=")[0].lstrip("-")
        for arg in argv[1:]
        if arg.startswith("--")
    }

    unknown = passed_flags - valid_params
    if unknown:
        raise LLMException(f"Unknown arguments for '{command}': {unknown}")


class CLI:
    """CLI commands exposed through Fire."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Build the retrieval index for the local codebase."""
        if max_chunk_size <= 0 or max_chunk_size > 2000:
            raise LLMException("max_chunk_size field need to be positive and lower than 2000")
        index_files("./data/raw/", max_chunk_size)

    def search(self, query: str, k: int = 5) -> None:
        """Run retrieval for a single query and print the results."""
        found_files: list[MinimalSource] = []
        retrieve_chunks(query, found_files, k=k)
        
        result = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id=str(uuid.uuid4()),
                    question_str=query,
                    retrieved_sources=found_files,
                )
            ],
            k=k,
        )
        print(result.model_dump_json(indent=2))

    def search_dataset(self, dataset_path: str, save_directory: str, k: int = 5) -> None:
        """Run retrieval for every question in a dataset and save the outputs."""
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = RagDataset.model_validate(json.load(f))
        except Exception as e:
            raise LLMException(f"failed to read/parse file: {dataset_path}. Error: {e}")
            
        search_results: list[MinimalSearchResults] = []
        for item in tqdm(dataset.rag_questions, desc="Searching dataset"):
            found_files: list[MinimalSource] = []
            retrieve_chunks(item.question, found_files, k=k)
            search_results.append(
                MinimalSearchResults(
                    question_id=item.question_id,
                    question_str=item.question,
                    retrieved_sources=found_files,
                )
            )

        result = StudentSearchResults(search_results=search_results, k=k)
        os.makedirs(save_directory, exist_ok=True)
        output_file_path = os.path.join(save_directory, os.path.basename(dataset_path))
        
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
        except Exception as e:
            raise LLMException(f"failed to write file: {output_file_path}") from e
            
        log_message(f"Saved student_search_results to {output_file_path}!", LogType.SUCCESS)

    def answer(self, query: str, k: int = 5) -> None:
        """Generate an answer for a single query using RAG."""
        model = AI()
        answer_text, t = model.RAG(query, k=k)

        # AI.RAG modifies found_files inside, but we need them here too. 
        # So we just run a quick retrieve_chunks purely to get the formatted MinimalSources
        found_files: list[MinimalSource] = []
        retrieve_chunks(query, found_files, k=k)

        result = StudentSearchResultsAndAnswer(
            search_results=[
                MinimalAnswer(
                    question_id=str(uuid.uuid4()),
                    question_str=query,
                    retrieved_sources=found_files,
                    answer=answer_text,
                )
            ],
            k=k,
        )
        print(result.model_dump_json(indent=2))
        log_message(f"Answered in {t:.2f}s", LogType.INFO)

    def answer_dataset(self, student_search_results_path: str, save_directory: str) -> None:
        """Generate answers for a saved search-results dataset and write them out."""
        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                search_data = StudentSearchResults.model_validate(json.load(f))
        except Exception as e:
            raise LLMException(f"failed to read/parse file: {student_search_results_path}. Error: {e}")

        model = AI()
        answers: list[MinimalAnswer] = []
        log_message(f"Loaded {len(search_data.search_results)} questions", LogType.INFO)

        for result in tqdm(search_data.search_results, desc="Generating answers"):
            answer_text, _ = model.RAG(result.question, k=search_data.k)
            answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question_str=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer_text,
                )
            )

        final_result = StudentSearchResultsAndAnswer(search_results=answers, k=search_data.k)
        os.makedirs(save_directory, exist_ok=True)
        output_file_path = os.path.join(save_directory, os.path.basename(student_search_results_path))
        
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(final_result.model_dump_json(indent=2))
        except Exception as e:
            raise LLMException(f"failed to write file: {output_file_path}") from e
            
        log_message(f"Saved outputs to {output_file_path}!", LogType.SUCCESS)

    def evaluate(self, student_answer_path: str | None = None, dataset_path: str | None = None, k: int = 10) -> None:
        """Evaluate retrieval quality against the bundled datasets."""
        if student_answer_path and dataset_path:
            evaluate_student_search_results(student_answer_path, dataset_path, k)
        else:
            evaluate("docs", 5)
            evaluate("code", 5)

def main() -> None:
    """Dispatch CLI commands."""
    cli = CLI()
    commands: CommandMap = {
        "index": cli.index,
        "search": cli.search,
        "search_dataset": cli.search_dataset,
        "answer": cli.answer,
        "answer_dataset": cli.answer_dataset,
        "evaluate": cli.evaluate,
    }
    validate_args(commands)
    fire_module: Any = fire
    fire_module.Fire(commands)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        log_message("Stopped the program.", LogType.WARNING)
    except LLMException as e:
        e.pretty_print()
    except FileNotFoundError as e:
        log_message(f"Could not access file: {e}", LogType.ERROR)
    except Exception as e:
        log_message(f"An error occurred: {e}", LogType.ERROR)