"""
Abstract base class for metric group evaluators.

Every metric group (VQA, captioning, hallucination, …) subclasses
:class:`BaseMetricEvaluator` and implements two things:

1. :pyattr:`group_name` — the key used in the snapshot ``metrics`` dict
   (e.g. ``"vqa"``, ``"captioning"``).
2. :meth:`evaluate` — runs inference on a dataset and returns a dict of
   metric scores that conforms to the snapshot schema for that group.

The base class provides shared helpers for dataset loading and progress
logging so that evaluators can focus on metric-specific logic.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BaseMetricEvaluator(ABC):
    """Abstract base for all metric group evaluators.

    Subclasses must implement:

    * :pyattr:`group_name` — schema key for the metric group.
    * :meth:`evaluate` — run the evaluation and return a metrics dict.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def group_name(self) -> str:
        """Return the schema group key (e.g. ``'vqa'``, ``'captioning'``)."""
        ...

    @abstractmethod
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
        """Run evaluation and return a metrics dict matching the schema.

        Args:
            model: A HuggingFace-compatible VLM (must support
                ``model.generate()``).
            processor: The paired processor / tokenizer that handles
                image + text inputs.
            device: Torch device string (``"cuda"``, ``"cpu"``, etc.).
            max_samples: If set, evaluate only this many samples (useful
                for quick testing).
            batch_size: Inference batch size.
            max_new_tokens: Maximum tokens to generate per sample.
            **kwargs: Evaluator-specific options.

        Returns:
            A dictionary conforming to the snapshot schema for this
            metric group.
        """
        ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_start(self, num_samples: int) -> None:
        """Log that evaluation is starting."""
        logger.info(
            "[%s] Starting evaluation on %d sample(s)…",
            self.group_name,
            num_samples,
        )

    def _log_done(self, results: Dict[str, Any]) -> None:
        """Log completion with primary metric value."""
        logger.info(
            "[%s] Evaluation complete — %s",
            self.group_name,
            {k: v for k, v in results.items()
             if isinstance(v, (int, float)) and v is not None},
        )
