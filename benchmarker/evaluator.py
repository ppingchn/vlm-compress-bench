"""
Evaluator orchestrator for VLM compression benchmarking.

This module provides the high-level API for running model evaluations.
It coordinates the individual metric group evaluators and assembles
the results into a complete snapshot dictionary.

Typical usage::

    from benchmarker.evaluator import full_evaluation

    snap = full_evaluation(
        model_name="LLaVA-1.5-7B",
        model=model,
        processor=processor,
        label="Baseline (FP16)",
        is_baseline=True,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from .metrics.base import BaseMetricEvaluator
from .metrics.general import evaluate_general_efficiency
from .metrics.vqa import VQAEvaluator
from .metrics.captioning import CaptioningEvaluator
from .metrics.visual_reasoning import VisualReasoningEvaluator
from .metrics.ocr_document import OCRDocumentEvaluator
from .metrics.hallucination import HallucinationEvaluator
from .metrics.spatial_awareness import SpatialAwarenessEvaluator
from .snapshot import snapshot as create_snapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry of all metric group evaluators
# ---------------------------------------------------------------------------

#: Ordered list of all available metric group evaluator classes.
ALL_EVALUATORS: List[type[BaseMetricEvaluator]] = [
    VQAEvaluator,
    CaptioningEvaluator,
    VisualReasoningEvaluator,
    OCRDocumentEvaluator,
    HallucinationEvaluator,
    SpatialAwarenessEvaluator,
]

#: Mapping from group name to evaluator class for quick lookup.
EVALUATOR_REGISTRY: Dict[str, type[BaseMetricEvaluator]] = {
    cls().group_name: cls for cls in ALL_EVALUATORS
}

#: All available group names.
ALL_GROUPS: List[str] = list(EVALUATOR_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    processor: Any,
    *,
    device: str = "cuda",
    groups: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 1,
    max_new_tokens: int = 128,
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """Run metric evaluations for the specified (or all) groups.

    Each group evaluator is instantiated and run sequentially.  Results
    are collected into a nested dict keyed by group name.

    Args:
        model: A HuggingFace-compatible VLM with ``model.generate()``.
        processor: The paired processor / tokenizer.
        device: Torch device string (``"cuda"``, ``"cpu"``).
        groups: List of group names to evaluate.  If ``None``, all six
            groups are evaluated.  Valid names:
            ``["vqa", "captioning", "visual_reasoning", "ocr_document",
            "hallucination", "spatial_awareness"]``.
        max_samples: Limit the number of samples per group (useful for
            quick testing / debugging).
        batch_size: Inference batch size passed to evaluators.
        max_new_tokens: Maximum tokens to generate per sample.
        **kwargs: Additional keyword arguments forwarded to each
            evaluator's ``evaluate()`` method.  Group-specific kwargs
            can be passed as ``vqa_dataset_name``, ``captioning_split``,
            etc. — prefixed kwargs are stripped of the prefix and
            forwarded to the matching evaluator.

    Returns:
        A nested dict ``{group_name: {metric_name: value}}``.

    Raises:
        ValueError: If an unknown group name is specified.

    Example::

        metrics = evaluate_model(
            model, processor,
            groups=["vqa", "captioning"],
            max_samples=100,
        )
    """
    if groups is None:
        groups = ALL_GROUPS

    # Validate group names
    unknown = set(groups) - set(ALL_GROUPS)
    if unknown:
        raise ValueError(
            f"Unknown metric group(s): {unknown}. "
            f"Available: {ALL_GROUPS}"
        )

    results: Dict[str, Dict[str, Any]] = {}

    for group_name in groups:
        evaluator_cls = EVALUATOR_REGISTRY[group_name]
        evaluator = evaluator_cls()

        # Extract group-specific kwargs (e.g. vqa_dataset_name → dataset_name)
        prefix = f"{group_name}_"
        group_kwargs: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if key.startswith(prefix):
                group_kwargs[key[len(prefix):]] = value

        logger.info("Running evaluation for group: %s", group_name)

        try:
            result = evaluator.evaluate(
                model,
                processor,
                device=device,
                max_samples=max_samples,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                **group_kwargs,
            )
            results[group_name] = result
            logger.info(
                "Group '%s' completed — %d samples evaluated.",
                group_name,
                result.get("num_samples", 0),
            )
        except Exception:
            logger.exception(
                "Group '%s' failed — skipping.", group_name,
            )

    return results


def full_evaluation(
    model_name: str,
    model: Any,
    processor: Any,
    label: str,
    *,
    model_info: Optional[dict] = None,
    device: str = "cuda",
    groups: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 1,
    max_new_tokens: int = 128,
    is_baseline: bool = False,
    baseline_size_mb: Optional[float] = None,
    sample_image: Any = None,
    sample_prompt: str = "Describe this image.",
    notes: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run a complete evaluation and return a ready-to-save snapshot.

    This is the main entry point for end-to-end benchmarking.  It:

    1. Measures general efficiency metrics (size, latency, memory, etc.)
    2. Runs all (or selected) capability metric groups.
    3. Assembles everything into a validated snapshot dictionary.

    Args:
        model_name: Name of the model (e.g. ``"LLaVA-1.5-7B"``).
        model: The VLM model object.
        processor: The paired processor / tokenizer.
        label: Human-readable label (e.g. ``"Baseline (FP16)"``).
        model_info: Model metadata dict.  If ``None``, a minimal dict is
            constructed from *model_name* and *is_baseline*.  Should
            contain keys like ``model_family``, ``base_llm``,
            ``vision_encoder``, ``precision``, ``compression_method``,
            ``compression_config``.
        device: Torch device string.
        groups: Which metric groups to evaluate (default: all).
        max_samples: Limit samples per group.
        batch_size: Inference batch size.
        max_new_tokens: Maximum generation tokens.
        is_baseline: Whether this is the baseline (reference) model.
        baseline_size_mb: Size of the baseline model in MB (for
            compression ratio).  If ``None`` and *is_baseline* is True,
            the model's own size is used.
        sample_image: A sample PIL Image for efficiency measurement.
            If ``None``, efficiency metrics that require inference are
            set to 0.
        sample_prompt: Prompt to use for efficiency measurement.
        notes: Optional freeform notes about this evaluation.
        **kwargs: Forwarded to :func:`evaluate_model`.

    Returns:
        A validated snapshot dictionary ready for :func:`save_snapshot`.
    """
    logger.info("Starting full evaluation for '%s'…", model_name)

    # ------------------------------------------------------------------
    # 1. General efficiency
    # ------------------------------------------------------------------
    if sample_image is not None:
        general_efficiency = evaluate_general_efficiency(
            model,
            processor,
            image=sample_image,
            prompt=sample_prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            baseline_size_mb=baseline_size_mb,
        )
    else:
        # Cannot measure latency / throughput without a sample image
        from .metrics.general import measure_model_size, count_parameters
        size_mb = measure_model_size(model)
        general_efficiency = {
            "model_size_mb": size_mb,
            "parameter_count": count_parameters(model),
            "compression_ratio": (
                (baseline_size_mb / size_mb) if baseline_size_mb and size_mb > 0
                else 1.0
            ),
            "inference_latency_ms": 0.0,
            "gpu_memory_mb": 0.0,
            "throughput_tokens_per_sec": 0.0,
        }
        logger.warning(
            "No sample_image provided — inference-based efficiency "
            "metrics (latency, GPU memory, throughput) set to 0."
        )

    # ------------------------------------------------------------------
    # 2. Capability metrics
    # ------------------------------------------------------------------
    metrics = evaluate_model(
        model,
        processor,
        device=device,
        groups=groups,
        max_samples=max_samples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )

    # ------------------------------------------------------------------
    # 3. Model info
    # ------------------------------------------------------------------
    if model_info is None:
        model_info = {
            "model_family": "",
            "base_llm": "",
            "vision_encoder": "",
            "precision": "",
            "compression_method": None,
            "compression_config": None,
        }

    # ------------------------------------------------------------------
    # 4. Evaluation config
    # ------------------------------------------------------------------
    evaluation_config = {
        "device": device,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "seed": None,
    }

    # ------------------------------------------------------------------
    # 5. Assemble snapshot
    # ------------------------------------------------------------------
    snap = create_snapshot(
        model_name=model_name,
        label=label,
        model_info=model_info,
        general_efficiency=general_efficiency,
        metrics=metrics,
        evaluation_config=evaluation_config,
        notes=notes,
        is_baseline=is_baseline,
    )

    logger.info(
        "Full evaluation complete for '%s' — snapshot ID: %s",
        model_name,
        snap["snapshot_id"],
    )
    return snap
