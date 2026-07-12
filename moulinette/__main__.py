import os
import json
from typing import Any

import fire

from src.models import (RagDataset,
                        StudentSearchResults,
                        AnsweredQuestion,
                        MinimalSource)


def normalize_path(path: str) -> str:
    """Strips any root codebase directory prefixes
    so paths conform to validation formats."""
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
        # Using <= 0 just to be mathematically safe against malformed spans
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

        # Check each expected source to see if it exists in the top k retrieved
        for expected in expected_sources:
            for retrieved in top_k_retrieved:
                overlap = self._overlap_proccess(retrieved, expected)
                if overlap >= 0.05:
                    sources_found += 1
                    # Found expected source; move to the next one.
                    break

        # Question score = number_found / total number of correct sources.
        return sources_found / len(expected_sources)


class MoulinetteCLI:
    """Calculates Recall@k"""

    def evaluate_student_search_results(
        self,
        student_answer_path: str,
        dataset_path: str,
        k: int = 10,
        max_context_length: int = 2000,
    ) -> None:
        """Calculate Recall@1, 3, 5, and 10 with the 5% overlap rule."""
        _ = max_context_length
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset path missing: {dataset_path}")
            return
        if not os.path.exists(student_answer_path):
            print(
                "Error: Student answer path missing: "
                f"{student_answer_path}"
            )
            return

        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_dataset: Any = json.load(f)

        with open(student_answer_path, "r", encoding="utf-8") as f:
            raw_student_data: Any = json.load(f)

        try:
            dataset = RagDataset.model_validate(raw_dataset)
            student_search = StudentSearchResults.model_validate(
                raw_student_data
            )
        except Exception as e:
            print(
                "Error validating JSON against Pydantic schema: "
                f"{e}"
            )
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

        # Calculate Macro-Average Recall@K
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

                # Calculate the score for this specific question
                q_score = recall_evaluator.calculate_question_recall(
                    retrieved_sources=retrieved_list,
                    expected_sources=expected_list,
                    k=c,
                )
                total_recall_score += q_score

            # Average the scores across all valid questions
            final_macro_recall = (
                total_recall_score / total_with_sources
                if total_with_sources > 0
                else 0.0
            )
            print(
                "Recall@"
                f"{c}: {final_macro_recall:.3f} "
                f"({(final_macro_recall * 100):.1f}%)"
            )


if __name__ == "__main__":
    fire.Fire(MoulinetteCLI)
