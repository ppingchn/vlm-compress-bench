"""
VQA (Visual Question Answering) metric evaluator.

This module evaluates Visual Question Answering performance using the VQA-v2
benchmark dataset. It computes overall soft accuracy as well as per-answer-type
accuracies (yes/no, number, other) following the official VQA-v2 evaluation
protocol.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from benchmarker.metrics.base import BaseMetricEvaluator
from benchmarker.utils import generate_text, normalize_answer

logger = logging.getLogger(__name__)


def vqa_soft_accuracy(prediction: str, ground_truths: list[str]) -> float:
    """Compute VQA-v2 soft accuracy score.

    The official VQA-v2 soft accuracy formula is:
        score = min(count(normalized_pred in normalized_gts) / 3, 1.0)

    Args:
        prediction: The model's predicted answer string.
        ground_truths: A list of ground-truth answer strings from human annotators.

    Returns:
        Soft accuracy score in the range [0.0, 1.0].
    """
    norm_pred = normalize_answer(prediction)
    norm_gts: list[str] = []
    for gt in ground_truths:
        if isinstance(gt, dict):
            gt_str = gt.get("answer", "")
        else:
            gt_str = str(gt)
        norm_gts.append(normalize_answer(gt_str))

    match_count = norm_gts.count(norm_pred)
    return min(match_count / 3.0, 1.0)


class VQAEvaluator(BaseMetricEvaluator):
    """Evaluator for Visual Question Answering (VQA-v2) performance.

    Computes overall soft accuracy and optional breakdown accuracies across
    question/answer categories (yes/no, number, other) using the VQA-v2
    soft accuracy metric.
    """

    @property
    def group_name(self) -> str:
        """Return the schema group key for VQA."""
        return "vqa"

    def evaluate(
        self,
        model: Any,
        processor: Any,
        *,
        device: str = "cuda",
        max_samples: Optional[int] = None,
        batch_size: int = 1,
        max_new_tokens: int = 128,
        dataset_name: str = "HuggingFaceM4/VQAv2",
        dataset: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Evaluate a VLM model on the VQA-v2 benchmark dataset.

        Args:
            model: A HuggingFace-compatible VLM model with generate() support.
            processor: The paired processor / tokenizer for handling images and text.
            device: Torch device string (e.g. "cuda", "cpu").
            max_samples: Optional limit on the number of samples to evaluate.
            batch_size: Batch size for evaluation (currently processes sample-by-sample).
            max_new_tokens: Maximum new tokens to generate per response.
            dataset_name: HuggingFace dataset identifier if `dataset` is not provided.
            dataset: Pre-loaded dataset or iterable of samples.
            **kwargs: Additional evaluator-specific keyword arguments.

        Returns:
            A dictionary conforming to the VQA metric snapshot schema:
            {
                "accuracy": float,
                "yes_no_accuracy": float | None,
                "number_accuracy": float | None,
                "other_accuracy": float | None,
                "dataset": str,
                "num_samples": int,
            }
        """
        if dataset is None:
            from datasets import load_dataset
            logger.info("Loading dataset '%s' (validation split)…", dataset_name)
            dataset = load_dataset(dataset_name, split="validation")

        eval_samples = dataset
        if max_samples is not None and max_samples > 0:
            if hasattr(eval_samples, "select") and hasattr(eval_samples, "__len__"):
                eval_samples = eval_samples.select(range(min(len(eval_samples), max_samples)))
            elif hasattr(eval_samples, "__len__"):
                eval_samples = eval_samples[:max_samples]

        total_samples = len(eval_samples) if hasattr(eval_samples, "__len__") else 0
        self._log_start(total_samples)

        total_scores: List[float] = []
        yes_no_scores: List[float] = []
        number_scores: List[float] = []
        other_scores: List[float] = []

        for idx, sample in enumerate(tqdm(eval_samples, desc="Evaluating VQA", total=total_samples)):
            if max_samples is not None and len(total_scores) >= max_samples:
                break

            try:
                image = sample.get("image")
                if image is None:
                    logger.warning("Sample %d missing 'image' field, skipping.", idx)
                    continue

                question = sample.get("question", "")
                if not question:
                    logger.warning("Sample %d missing 'question' field, skipping.", idx)
                    continue

                raw_answers = sample.get("answers", [])
                if isinstance(raw_answers, list):
                    ground_truths = []
                    for ans in raw_answers:
                        if isinstance(ans, dict):
                            ground_truths.append(ans.get("answer", ""))
                        else:
                            ground_truths.append(str(ans))
                elif isinstance(raw_answers, str):
                    ground_truths = [raw_answers]
                else:
                    ground_truths = []

                if not ground_truths:
                    if "multiple_choice_answer" in sample and sample["multiple_choice_answer"]:
                        ground_truths = [str(sample["multiple_choice_answer"])]
                    elif "answer" in sample and sample["answer"]:
                        ground_truths = [str(sample["answer"])]

                prompt = (
                    f"Question: {question}\n"
                    "Answer the question using a single word or phrase."
                )

                prediction = generate_text(
                    model,
                    processor,
                    image=image,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=max_new_tokens,
                )

                score = vqa_soft_accuracy(prediction, ground_truths)
                total_scores.append(score)

                raw_atype = sample.get("answer_type") or sample.get("question_type")
                if raw_atype is not None and isinstance(raw_atype, str):
                    atype_clean = raw_atype.lower().strip().replace("-", "_").replace("/", "_")
                    if "yes" in atype_clean or "no" in atype_clean:
                        yes_no_scores.append(score)
                    elif "number" in atype_clean or "num" in atype_clean:
                        number_scores.append(score)
                    elif "other" in atype_clean:
                        other_scores.append(score)

            except Exception as e:
                logger.warning("Error evaluating sample %d: %s", idx, e)
                continue

        num_evaluated = len(total_scores)
        overall_acc = float(sum(total_scores) / num_evaluated) if num_evaluated > 0 else 0.0
        yes_no_acc = (
            float(sum(yes_no_scores) / len(yes_no_scores))
            if yes_no_scores
            else None
        )
        number_acc = (
            float(sum(number_scores) / len(number_scores))
            if number_scores
            else None
        )
        other_acc = (
            float(sum(other_scores) / len(other_scores))
            if other_scores
            else None
        )

        dataset_label = "VQA-v2" if dataset_name == "HuggingFaceM4/VQAv2" else dataset_name

        results = {
            "accuracy": overall_acc,
            "yes_no_accuracy": yes_no_acc,
            "number_accuracy": number_acc,
            "other_accuracy": other_acc,
            "dataset": dataset_label,
            "num_samples": num_evaluated,
        }

        self._log_done(results)
        return results
