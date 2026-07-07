import os
import json
from typing import Any

import fire  # type: ignore

from src.models.models import (
    RagDataset,
    StudentSearchResults,
    AnsweredQuestion,
    MinimalSource
)

def normalize_path(path: str) -> str:
    """Strips any root codebase directory prefixes so paths conform to validation formats."""
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
        self,
        retrieved: MinimalSource,
        expected: MinimalSource
    ) -> float:
        """Compute normalized overlap between retrieved and expected spans."""
        if normalize_path(retrieved.file_path) != normalize_path(expected.file_path):
            return 0.0
            
        start_inter = max(
            retrieved.first_character_index,
            expected.first_character_index
        )
        end_inter = min(
            retrieved.last_character_index,
            expected.last_character_index
        )
        overlap_length = max(0, end_inter - start_inter)
        expected_length = (
            expected.last_character_index -
            expected.first_character_index
        )
        
        # Condition threshold match check (>= 5% overlap metric)
        if expected_length == 0:
            return 0.0
            
        return overlap_length / expected_length

    def calculate(
        self, 
        retrieved_sources: list[list[MinimalSource]],
        expected_sources: list[MinimalSource], 
        k: int
    ) -> float:
        """Calculate recall at `k` using minimum overlap threshold."""
        sources_found = 0
        for idx, expct in enumerate(expected_sources):
            if idx >= len(retrieved_sources):
                continue
                
            top_retrieved_sources = retrieved_sources[idx][:k]

            for retriev in top_retrieved_sources:
                overlap = self._overlap_proccess(retriev, expct)
                if overlap >= 0.05:
                    sources_found += 1
                    break

        if not expected_sources:
            return 0.0

        recall_score = sources_found / len(expected_sources)
        return recall_score


class MoulinetteCLI:
    """Calculates Recall@k"""

    def evaluate_student_search_results(
        self,
        student_answer_path: str,
        dataset_path: str,
        k: int = 10,
        max_context_length: int = 2000
    ) -> None:
        """
        Calculates Recall@1, 3, 5, and 10 based on the 5% overlap rule.
        """
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset path missing: {dataset_path}")
            return
        if not os.path.exists(student_answer_path):
            print(f"Error: Student answer path missing: {student_answer_path}")
            return

        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_dataset: Any = json.load(f)
            
        with open(student_answer_path, "r", encoding="utf-8") as f:
            raw_student_data: Any = json.load(f)

        try:
            dataset = RagDataset.model_validate(raw_dataset)
            student_search = StudentSearchResults.model_validate(raw_student_data)
        except Exception as e:
            print(f"Error validating JSON against Pydantic schema: {e}")
            return

        print("Student data is valid: True")
        
        student_search_map = {
            res.question_id: res for res in student_search.search_results
        }
        
        valid_gt_questions = [
            q for q in dataset.rag_questions 
            if isinstance(q, AnsweredQuestion) and len(q.sources) > 0
        ]
        
        total_questions = len(dataset.rag_questions)
        total_with_sources = len(valid_gt_questions)
        
        questions_with_student_sources = sum(
            1 for q in valid_gt_questions 
            if q.question_id in student_search_map 
            and len(student_search_map[q.question_id].retrieved_sources) > 0
        )

        print(f"Total number of questions: {total_questions}")
        print(f"Total number of questions with sources: {total_with_sources}")
        print(f"Total number of questions with student sources: {questions_with_student_sources}")

        flat_expected: list[MinimalSource] = []
        flat_retrieved: list[list[MinimalSource]] = []

        for q in valid_gt_questions:
            student_res = student_search_map.get(q.question_id)
            retrieved_list = student_res.retrieved_sources if student_res else []
            
            for expct in q.sources:
                flat_expected.append(expct)
                flat_retrieved.append(retrieved_list)

        recall_evaluator = Recall()
        cutoffs = [1, 3, 5, 10]
        
        print("Evaluation Results")
        print("========================================")
        print(f"Questions evaluated: {total_with_sources}")

        for c in cutoffs:
            score = recall_evaluator.calculate(flat_retrieved, flat_expected, c)
            print(f"Recall@{c}: {score:.3f}")


if __name__ == "__main__":
    fire.Fire(MoulinetteCLI)