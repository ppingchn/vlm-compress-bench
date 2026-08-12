"""
OCR and Document Understanding evaluator using TextVQA and ANLS metric.

This module evaluates Visual Language Models (VLMs) on reading comprehension
and OCR tasks. Evaluation is performed on the TextVQA dataset using Average
Normalized Levenshtein Similarity (ANLS) as the primary metric, alongside
exact match accuracy and partial match rate as supporting metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from benchmarker.metrics.base import BaseMetricEvaluator
from benchmarker.utils import anls_score, generate_text, normalize_answer

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Answer the question based on the text in the image. Use a short phrase."
)


class OCRDocumentEvaluator(BaseMetricEvaluator):
    """Evaluator for OCR and Document Understanding capability group.

    Evaluates VLM performance on TextVQA validation set using ANLS,
    exact match accuracy, and partial match rate (> 0.5 ANLS).
    """

    @property
    def group_name(self) -> str:
        """Return the schema group key for OCR/Document understanding."""
        return "ocr_document"

    def evaluate(
        self,
        model: Any,
        processor: Any,
        *,
        device: str = "cuda",
        max_samples: Optional[int] = None,
        batch_size: int = 1,
        max_new_tokens: int = 128,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run OCR / Document Understanding evaluation on TextVQA.

        Args:
            model: HuggingFace VLM with ``generate()`` support.
            processor: Processor/tokenizer handling images and text.
            device: Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
            max_samples: Limit evaluation to this number of samples if set.
            batch_size: Inference batch size (defaults to 1).
            max_new_tokens: Maximum tokens to generate per prompt.
            **kwargs: Additional evaluation options.
                - ``dataset_name`` (str): HuggingFace dataset name
                  (default: ``"facebook/textvqa"``).
                - ``dataset`` (iterable): Pre-loaded dataset split.

        Returns:
            Dictionary matching the ``ocr_document`` schema group:
            - ``anls`` (float): Mean ANLS score across valid samples.
            - ``exact_match_accuracy`` (float | None): Proportion of exact matches.
            - ``partial_match_rate`` (float | None): Proportion with ANLS > 0.5.
            - ``dataset`` (str): Dataset name ("TextVQA").
            - ``num_samples`` (int): Number of valid samples evaluated.
        """
        dataset = kwargs.get("dataset")
        dataset_name = kwargs.get("dataset_name", "facebook/textvqa")

        if dataset is None:
            try:
                from datasets import load_dataset

                logger.info("Loading dataset '%s' (validation split)...", dataset_name)
                dataset = load_dataset(dataset_name, split="validation")
            except Exception as err:
                logger.error("Failed to load dataset '%s': %s", dataset_name, err)
                raise RuntimeError(
                    f"Failed to load dataset '{dataset_name}' for OCR evaluation: {err}"
                ) from err

        # Determine number of samples to process
        total_samples = len(dataset)
        if max_samples is not None and max_samples > 0:
            total_samples = min(total_samples, max_samples)

        self._log_start(total_samples)

        anls_scores: List[float] = []
        exact_matches: List[bool] = []
        partial_matches: List[bool] = []

        # Iterate over samples up to total_samples
        for i in tqdm(range(total_samples), desc="Evaluating OCR/Document (TextVQA)"):
            try:
                sample = dataset[i]
                question = sample["question"]
                raw_answers = (
                    sample.get("answers")
                    or sample.get("answers_list")
                    or sample.get("ground_truth")
                    or []
                )

                if isinstance(raw_answers, str):
                    raw_answers = [raw_answers]

                gt_norms = [normalize_answer(gt) for gt in raw_answers if gt]

                image = sample.get("image")
                if hasattr(image, "convert"):
                    image = image.convert("RGB")

                prompt = PROMPT_TEMPLATE.format(question=question)

                pred_raw = generate_text(
                    model,
                    processor,
                    image=image,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=max_new_tokens,
                )

                pred_norm = normalize_answer(pred_raw)

                # Compute ANLS score (max over ground truth answers)
                score = anls_score(pred_norm, gt_norms, threshold=0.5)
                anls_scores.append(score)

                # Exact match if prediction matches any ground-truth answer
                is_exact = any(pred_norm == gt for gt in gt_norms) if gt_norms else False
                exact_matches.append(is_exact)

                # Partial match if ANLS > 0.5
                is_partial = score > 0.5
                partial_matches.append(is_partial)

            except Exception as err:
                logger.warning("Error evaluating sample index %d: %s. Skipping.", i, err)
                continue

        num_valid = len(anls_scores)
        if num_valid == 0:
            logger.warning("No samples were successfully evaluated.")
            results = {
                "anls": 0.0,
                "exact_match_accuracy": None,
                "partial_match_rate": None,
                "dataset": "TextVQA",
                "num_samples": 0,
            }
        else:
            mean_anls = sum(anls_scores) / num_valid
            exact_match_acc = sum(exact_matches) / num_valid
            partial_match_rate = sum(partial_matches) / num_valid

            results = {
                "anls": float(mean_anls),
                "exact_match_accuracy": float(exact_match_acc),
                "partial_match_rate": float(partial_match_rate),
                "dataset": "TextVQA",
                "num_samples": num_valid,
            }

        self._log_done(results)
        return results


__all__ = ["OCRDocumentEvaluator"]
