"""Command-line entry point for indexing, retrieval, and answer generation."""

import inspect
import sys
import json
import os
import uuid
import fire
from collections.abc import Callable
from typing import Any
from tqdm import tqdm

from src.indexer import index_files, retrieve_chunks
from src.exceptions import LLMException, LogType, log_message
from src.models import (
    MinimalSource,
    StudentSearchResults,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResultsAndAnswer,
    MinimalAnswer,
    AnsweredQuestion,
)
from src.ai import AI

CommandMap = dict[str, Callable[..., Any]]


def normalize_path(path: str) -> str:
    """Strip root codebase prefixes so paths conform to validation formats."""
    path = path.replace("\\", "/")
    prefixes = ["vllm-0.10.1/", "./vllm-0.10.1/"]
    for prefix in prefixes:
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


class Recall:
    """Compute retrieval recall from source span overlaps."""

    def __init__(self) -> None:
        """Initialize recall calculator."""
        pass

    def _overlap_proccess(
        self, retrieved: MinimalSource, expected: MinimalSource
    ) -> float:
        """Compute normalized overlap between retrieved and expected spans."""
        if (
            normalize_path(retrieved.file_path)
            != normalize_path(expected.file_path)
        ):
            return 0.0

        start_inter = max(
            retrieved.first_character_index, expected.first_character_index
        )
        end_inter = min(
            retrieved.last_character_index,
            expected.last_character_index,
        )

        overlap_length = max(0, end_inter - start_inter)
        expected_length = (
            expected.last_character_index - expected.first_character_index
        )

        # Condition threshold match check (>= 5% overlap metric)
        if expected_length <= 0:
            return 0.0

        return overlap_length / expected_length

    def calculate_question_recall(
        self,
        retrieved_sources: list[MinimalSource],
        expected_sources: list[MinimalSource],
        k: int,
    ) -> float:
        """Calculate Recall@k for one question using the overlap threshold."""
        if not expected_sources:
            return 0.0

        top_k_retrieved = retrieved_sources[:k]
        sources_found = 0

        for expected in expected_sources:
            for retrieved in top_k_retrieved:
                overlap = self._overlap_proccess(retrieved, expected)
                if overlap >= 0.05:
                    sources_found += 1
                    break

        return sources_found / len(expected_sources)


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
            raise LLMException(
                "max_chunk_size field need to be positive and lower than 2000"
            )
        index_files("./data/raw/", max_chunk_size)

    def search(
        self,
        query: str,
        k: int = 5,
        output_path: str = "data/output/search_output.json",
    ) -> None:
        """Run retrieval for a single query and save formatted results."""
        found_files: list[MinimalSource] = []
        retrieve_chunks(query, found_files, k=k)

        for source in found_files:
            clean_path = source.file_path.removeprefix("./")
            print(
                f"{clean_path} "
                f"[{source.first_character_index}:"
                f"{source.last_character_index}]"
            )

        result = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id=str(uuid.uuid4()),
                    question=query,
                    retrieved_sources=found_files,
                )
            ],
            k=k,
        )

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
        except Exception as e:
            raise LLMException(
                f"Failed to write file: {output_path}. Error: {e}"
            ) from e

    def search_dataset(
        self, dataset_path: str, save_directory: str, k: int = 5
    ) -> None:
        """Run retrieval over a dataset and save the retrieval outputs."""
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = RagDataset.model_validate(json.load(f))
        except Exception as e:
            raise LLMException(
                f"failed to read/parse file: {dataset_path}. Error: {e}"
            )

        search_results: list[MinimalSearchResults] = []
        for item in tqdm(dataset.rag_questions, desc="Searching dataset"):
            found_files: list[MinimalSource] = []
            retrieve_chunks(item.question, found_files, k=k)

            for source in found_files:
                source.file_path = source.file_path.removeprefix("./")

            search_results.append(
                MinimalSearchResults(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=found_files,
                )
            )

        result = StudentSearchResults(search_results=search_results, k=k)
        os.makedirs(save_directory, exist_ok=True)
        output_file_path = os.path.join(
            save_directory, os.path.basename(dataset_path)
        )

        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
        except Exception as e:
            raise LLMException(
                f"failed to write file: {output_file_path}"
            ) from e

        log_message(
            f"Saved student_search_results to {output_file_path}!",
            LogType.SUCCESS,
        )

    def answer(
        self,
        query: str,
        k: int = 5,
        output_path: str = "data/output/answer_output.json",
    ) -> None:
        """Generate an answer for a single query using RAG and save details."""
        model = AI()
        answer_text, t = model.RAG(query, k=k)

        print(f"\nAnswer:\n{answer_text}\n")

        found_files: list[MinimalSource] = []
        retrieve_chunks(query, found_files, k=k)

        for source in found_files:
            source.file_path = source.file_path.removeprefix("./")

        result = StudentSearchResultsAndAnswer(
            search_results=[
                MinimalAnswer(
                    question_id=str(uuid.uuid4()),
                    question=query,
                    retrieved_sources=found_files,
                    answer=answer_text,
                )
            ],
            k=k,
        )

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
        except Exception as e:
            raise LLMException(
                f"Failed to write file: {output_path}. Error: {e}"
            ) from e

        log_message(f"Answered in {t:.2f}s", LogType.INFO)

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Generate answers for saved search results and write outputs."""
        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                search_data = StudentSearchResults.model_validate(json.load(f))
        except Exception as e:
            raise LLMException(
                "failed to read/parse file: "
                f"{student_search_results_path}. Error: {e}"
            )

        model = AI()
        answers: list[MinimalAnswer] = []
        log_message(
            f"Loaded {len(search_data.search_results)} questions", LogType.INFO
        )

        for result in tqdm(
            search_data.search_results,
            desc="Generating answers",
        ):
            answer_text, _ = model.RAG(result.question, k=search_data.k)
            answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer_text,
                )
            )

        final_result = StudentSearchResultsAndAnswer(
            search_results=answers, k=search_data.k
        )
        os.makedirs(save_directory, exist_ok=True)
        output_file_path = os.path.join(
            save_directory, os.path.basename(student_search_results_path)
        )

        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(final_result.model_dump_json(indent=2))
        except Exception as e:
            raise LLMException(
                f"failed to write file: {output_file_path}"
            ) from e

        log_message(f"Saved outputs to {output_file_path}!", LogType.SUCCESS)

    def evaluate(
        self,
        student_search_results_path: Any = None,
        dataset_path: Any = None,
        k: int = 10,
    ) -> None:
        """Evaluate retrieval quality against the bundled datasets."""
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset path missing: {dataset_path}")
            return
        if not os.path.exists(student_search_results_path):
            print(
                "Error: Student answer path missing: "
                f"{student_search_results_path}"
            )
            return

        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_dataset: Any = json.load(f)

        with open(student_search_results_path, "r", encoding="utf-8") as f:
            raw_student_data: Any = json.load(f)

        try:
            dataset = RagDataset.model_validate(raw_dataset)
            student_search = StudentSearchResults.model_validate(
                raw_student_data
            )
        except Exception as e:
            print(f"Error validating JSON against Pydantic schema: {e}")
            return

        print("Student data is valid: True")

        student_search_map = {
            res.question_id: res for res in student_search.search_results
        }

        valid_gt_questions = [
            q
            for q in dataset.rag_questions
            if (
                isinstance(q, AnsweredQuestion)
                and q.sources is not None
                and len(q.sources) > 0
            )
        ]

        total_questions = len(dataset.rag_questions)
        total_with_sources = len(valid_gt_questions)

        questions_with_student_sources = sum(
            1
            for q in valid_gt_questions
            if q.question_id in student_search_map
            and len(student_search_map[q.question_id].retrieved_sources) > 0
        )

        print(f"Total number of questions: {total_questions}")
        print(f"Total number of questions with sources: {total_with_sources}")
        print(
            "Total number of questions with student sources: "
            f"{questions_with_student_sources}"
        )

        recall_evaluator = Recall()
        cutoffs = [1, 3, 5, 10]

        print("\nEvaluation Results")
        print("========================================")
        print(f"Questions evaluated: {total_with_sources}")

        for c in cutoffs:
            total_recall_score = 0.0

            for q in valid_gt_questions:
                if q.sources is None:
                    continue

                student_res = student_search_map.get(q.question_id)
                retrieved_list = (
                    student_res.retrieved_sources if student_res else []
                )
                expected_list = q.sources

                q_score = recall_evaluator.calculate_question_recall(
                    retrieved_sources=retrieved_list,
                    expected_sources=expected_list,
                    k=c,
                )
                total_recall_score += q_score

            final_macro_recall = (
                total_recall_score / total_with_sources
                if total_with_sources > 0
                else 0.0
            )
            print(
                f"Recall@{c}: {final_macro_recall:.3f} "
                f"({(final_macro_recall * 100):.1f}%)"
            )


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
