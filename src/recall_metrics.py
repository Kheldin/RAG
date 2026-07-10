"""Recall evaluation helpers for retrieval results."""

import os
import json
from tqdm import tqdm
from .exceptions import LLMException, LogType, log_message
from .indexer import retrieve_chunks
from .models.models import AnsweredQuestion, MinimalSource, StudentSearchResults

def evaluate(dataset_type: str, k: int = 3) -> None:
    """Evaluate retrieval recall against one of the public answer datasets."""
    dataset_path = f"./data/datasets/AnsweredQuestions/dataset_{dataset_type}_public.json"
    
    if not os.path.exists(dataset_path):
        log_message(f"Skipping auto-evaluate for {dataset_type}: '{dataset_path}' not found.", LogType.WARNING)
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        question_list = [AnsweredQuestion(**d) for d in data.get("rag_questions", [])]

    recalls = []
    for question in tqdm(question_list, desc=f"Evaluating {dataset_type}"):
        found_files: list[MinimalSource] = []
        retrieve_chunks(question.question, found_files, k=k)

        hits = 0
        if not question.sources:
            continue
            
        total_relevant = len(question.sources)
        for source in question.sources:
            gt_path = source.file_path if source.file_path.startswith("./") else f"./{source.file_path}"
            gt_start = source.first_character_index
            gt_end = source.last_character_index

            for found in found_files:
                pred_path = found.file_path if found.file_path.startswith("./") else f"./{found.file_path}"
                if pred_path != gt_path:
                    continue

                overlap = max(0, min(gt_end, found.last_character_index) - max(gt_start, found.first_character_index))
                gt_length = gt_end - gt_start

                if gt_length <= 0 or overlap / gt_length >= 0.05:
                    hits += 1
                    break

        recall = hits / total_relevant if total_relevant > 0 else 0
        recalls.append(recall)

    final_recall = sum(recalls) * 100 / len(recalls) if recalls else 0
    log_message(f"RAG evaluation:\nRecall@{k} ({dataset_type}) = {final_recall:.2f}%", LogType.INFO)

    if (dataset_type == "code" and final_recall < 45.0) or (dataset_type == "docs" and final_recall < 55.0):
        log_message(f"Recall@{k} for {dataset_type} has not been achieved!", LogType.WARNING)


def evaluate_student_search_results(student_answer_path: str, dataset_path: str, k: int) -> None:
    """Evaluate saved student search results against a ground-truth dataset."""
    if not os.path.exists(student_answer_path):
        raise LLMException(f"Student answer file not found: {student_answer_path}")
    if not os.path.exists(dataset_path):
        raise LLMException(f"Ground-truth dataset file not found: {dataset_path}")

    # Load Ground Truth
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        dataset_questions = {
            d["question_id"]: AnsweredQuestion(**d)
            for d in data.get("rag_questions", [])
            if d.get("sources")
        }

    # Load Student Predictions
    with open(student_answer_path, "r", encoding="utf-8") as f:
        student_data = json.load(f)
        try:
            search_results = StudentSearchResults.model_validate(student_data).search_results
        except Exception as e:
            raise LLMException(f"Invalid student answers format. Error: {e}")

    recalls = []
    for result in search_results:
        question_id = result.question_id
        if question_id not in dataset_questions:
            continue

        gt_question = dataset_questions[question_id]
        found_files = result.retrieved_sources[:k]

        hits = 0
        
        # Pydantic typing safety since we made sources Optional earlier
        if not gt_question.sources:
            continue
            
        total_relevant = len(gt_question.sources)
        for source in gt_question.sources:
            gt_path = source.file_path if source.file_path.startswith("./") else f"./{source.file_path}"
            gt_start = source.first_character_index
            gt_end = source.last_character_index

            for found in found_files:
                pred_path = found.file_path if found.file_path.startswith("./") else f"./{found.file_path}"
                
                if pred_path != gt_path:
                    continue

                overlap = max(0, min(gt_end, found.last_character_index) - max(gt_start, found.first_character_index))
                gt_length = gt_end - gt_start

                if gt_length <= 0 or overlap / gt_length >= 0.05:
                    hits += 1
                    break

        recall = hits / total_relevant if total_relevant > 0 else 0
        recalls.append(recall)

    if not recalls:
        log_message("No matching questions found between the student results and the dataset.", LogType.WARNING)
        return

    final_recall = sum(recalls) * 100 / len(recalls)
    log_message(f"RAG evaluation:\nRecall@{k} = {final_recall:.2f}%", LogType.SUCCESS)