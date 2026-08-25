"""Spatial Awareness metric evaluator using the VSR (Visual Spatial Reasoning) dataset.

This module evaluates a Visual Language Model's (VLM) spatial awareness by asking
binary true/false questions about spatial relations between objects in images.
Spatial relations are categorized into vertical, horizontal, proximity, and
containment types.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

from PIL import Image
from tqdm import tqdm

from benchmarker.metrics.base import BaseMetricEvaluator
from benchmarker.utils import extract_true_false, generate_text

logger = logging.getLogger(__name__)

#: Sets of spatial relation labels mapped to high-level spatial categories.
VERTICAL_RELATIONS: Set[str] = {
    "above",
    "below",
    "under",
    "over",
    "beneath",
    "on top of",
}

HORIZONTAL_RELATIONS: Set[str] = {
    "left of",
    "right of",
    "to the left of",
    "to the right of",
}

PROXIMITY_RELATIONS: Set[str] = {
    "near",
    "far from",
    "close to",
    "far away from",
}

CONTAINMENT_RELATIONS: Set[str] = {
    "inside",
    "outside",
    "within",
    "in",
    "out of",
    "contains",
}


def classify_relation(relation_text: str) -> Optional[str]:
    """Classify a spatial relation text into a category.

    Categories:
    - 'vertical': relations such as 'above', 'below', 'under', 'over', 'beneath', 'on top of'.
    - 'horizontal': relations such as 'left of', 'right of', 'to the left of', 'to the right of'.
    - 'proximity': relations such as 'near', 'far from', 'close to', 'far away from'.
    - 'containment': relations such as 'inside', 'outside', 'within', 'in', 'out of', 'contains'.

    Args:
        relation_text: Spatial relation label from the dataset.

    Returns:
        One of 'vertical', 'horizontal', 'proximity', 'containment',
        or None if relation_text is empty, missing, or unrecognized.
    """
    if not relation_text or not isinstance(relation_text, str):
        return None
    rel = relation_text.strip().lower().replace("_", " ")
    if rel in VERTICAL_RELATIONS:
        return "vertical"
    if rel in HORIZONTAL_RELATIONS:
        return "horizontal"
    if rel in PROXIMITY_RELATIONS:
        return "proximity"
    if rel in CONTAINMENT_RELATIONS:
        return "containment"
    return None


class SpatialAwarenessEvaluator(BaseMetricEvaluator):
    """Evaluator for Spatial Awareness capability on the VSR dataset.

    Evaluates VLM performance on binary true/false visual spatial reasoning
    tasks. Outputs overall accuracy, sub-accuracies by spatial relation type
    (vertical, horizontal, proximity, containment), and performance gap
    above random chance baseline (0.5).
    """

    @property
    def group_name(self) -> str:
        """Return the schema group key for spatial awareness metrics."""
        return "spatial_awareness"

    def evaluate(
        self,
        model: Any,
        processor: Any,
        *,
        device: str = "cuda",
        max_samples: Optional[int] = None,
        batch_size: int = 1,
        max_new_tokens: int = 128,
        dataset_name: str = "cambridgeltl/vsr_random",
        dataset: Optional[Any] = None,
        split: str = "test",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Evaluate spatial awareness on the VSR dataset.

        Args:
            model: A HuggingFace-compatible VLM instance (must support ``model.generate()``).
            processor: Paired processor for image + text inputs.
            device: Torch device string (default: "cuda").
            max_samples: Maximum number of samples to evaluate (optional).
            batch_size: Inference batch size (default: 1).
            max_new_tokens: Maximum tokens to generate per response (default: 128).
            dataset_name: HuggingFace dataset identifier (default: "cambridgeltl/vsr_random").
            dataset: Pre-loaded dataset object (optional).
            split: Dataset split to evaluate (default: "test").
            **kwargs: Evaluator-specific options.

        Returns:
            Dictionary matching the spatial_awareness snapshot schema:
            - accuracy (float): Overall accuracy on VSR.
            - above_below_accuracy (float | None): Accuracy on vertical relations.
            - left_right_accuracy (float | None): Accuracy on horizontal relations.
            - near_far_accuracy (float | None): Accuracy on proximity relations.
            - inside_outside_accuracy (float | None): Accuracy on containment relations.
            - gap_above_random (float | None): Overall accuracy minus 0.5 random baseline.
            - dataset (str): Dataset name ("VSR").
            - num_samples (int): Total evaluated sample count.
        """
        if dataset is not None:
            ds = dataset
            if hasattr(ds, "keys") and split in ds:
                ds = ds[split]
        else:
            try:
                from datasets import load_dataset
                ds = load_dataset(dataset_name, split=split)
            except Exception as err:
                logger.error("Failed to load dataset '%s': %s", dataset_name, err)
                raise

        if max_samples is not None and max_samples > 0:
            if hasattr(ds, "select"):
                num_to_eval = min(len(ds), max_samples)
                ds = ds.select(range(num_to_eval))
            elif isinstance(ds, list):
                ds = ds[:max_samples]

        num_samples_to_eval = len(ds)
        self._log_start(num_samples_to_eval)

        total_correct = 0
        evaluated_samples = 0

        category_counts: Dict[str, Dict[str, int]] = {
            "vertical": {"correct": 0, "total": 0},
            "horizontal": {"correct": 0, "total": 0},
            "proximity": {"correct": 0, "total": 0},
            "containment": {"correct": 0, "total": 0},
        }

        for sample in tqdm(ds, desc="Evaluating Spatial Awareness (VSR)"):
            try:
                # Extract image
                image = sample.get("image")
                if image is None:
                    logger.warning("Sample missing 'image' field, skipping.")
                    continue

                if isinstance(image, str):
                    image = Image.open(image).convert("RGB")
                elif hasattr(image, "convert"):
                    image = image.convert("RGB")

                # Extract caption
                caption = sample.get("caption", "")

                # Construct prompt
                prompt = (
                    f"Is the following statement about the image true or false?\n"
                    f'Statement: "{caption}"\n'
                    f"Answer with only 'true' or 'false'."
                )

                # Generate model response
                response_text = generate_text(
                    model=model,
                    processor=processor,
                    image=image,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=max_new_tokens,
                )

                # Extract boolean prediction (returns True, False, or None if ambiguous)
                pred_bool = extract_true_false(response_text)

                # Ground truth parsing (0 = False, 1 = True)
                raw_label = sample.get("label")
                if isinstance(raw_label, bool):
                    gt_bool = raw_label
                elif isinstance(raw_label, (int, float)):
                    gt_bool = bool(int(raw_label) == 1)
                elif isinstance(raw_label, str):
                    gt_bool = raw_label.strip().lower() in ("1", "true")
                else:
                    gt_bool = bool(raw_label)

                # Count ambiguous model outputs (pred_bool is None) as incorrect
                is_correct = (pred_bool is not None) and (pred_bool == gt_bool)

                if is_correct:
                    total_correct += 1

                evaluated_samples += 1

                # Track relation sub-accuracies if relation field is present
                relation_text = sample.get("relation")
                if relation_text is not None:
                    cat = classify_relation(str(relation_text))
                    if cat in category_counts:
                        category_counts[cat]["total"] += 1
                        if is_correct:
                            category_counts[cat]["correct"] += 1

            except Exception as err:
                logger.warning("Failed to evaluate sample: %s", err, exc_info=True)
                continue

        accuracy = (total_correct / evaluated_samples) if evaluated_samples > 0 else 0.0

        above_below_acc = (
            category_counts["vertical"]["correct"] / category_counts["vertical"]["total"]
            if category_counts["vertical"]["total"] > 0
            else None
        )
        left_right_acc = (
            category_counts["horizontal"]["correct"] / category_counts["horizontal"]["total"]
            if category_counts["horizontal"]["total"] > 0
            else None
        )
        near_far_acc = (
            category_counts["proximity"]["correct"] / category_counts["proximity"]["total"]
            if category_counts["proximity"]["total"] > 0
            else None
        )
        inside_outside_acc = (
            category_counts["containment"]["correct"] / category_counts["containment"]["total"]
            if category_counts["containment"]["total"] > 0
            else None
        )

        gap_above_random = (accuracy - 0.5) if evaluated_samples > 0 else None

        results: Dict[str, Any] = {
            "accuracy": float(accuracy),
            "above_below_accuracy": float(above_below_acc) if above_below_acc is not None else None,
            "left_right_accuracy": float(left_right_acc) if left_right_acc is not None else None,
            "near_far_accuracy": float(near_far_acc) if near_far_acc is not None else None,
            "inside_outside_accuracy": float(inside_outside_acc) if inside_outside_acc is not None else None,
            "gap_above_random": float(gap_above_random) if gap_above_random is not None else None,
            "dataset": "VSR",
            "num_samples": evaluated_samples,
        }

        self._log_done(results)
        return results
