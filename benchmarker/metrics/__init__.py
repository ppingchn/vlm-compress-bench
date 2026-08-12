"""
Metric group implementations for VLM compression benchmarking.

This package contains individual evaluator modules for each of the six
capability metric groups, plus a general-efficiency measurement module
and an abstract base class.

Modules:
    - base.py              — Abstract base class for metric evaluators
    - general.py           — General efficiency metrics (size, latency, etc.)
    - vqa.py               — Visual Question Answering (VQA-v2)
    - captioning.py        — Image Captioning (COCO Captions, CIDEr)
    - visual_reasoning.py  — Visual Reasoning (GQA)
    - ocr_document.py      — OCR / Document Understanding (TextVQA, ANLS)
    - hallucination.py     — Hallucination detection (CHAIR-S, lower-is-better)
    - spatial_awareness.py — Spatial Awareness (VSR)
"""

from .base import BaseMetricEvaluator
from .general import evaluate_general_efficiency
from .vqa import VQAEvaluator
from .captioning import CaptioningEvaluator
from .visual_reasoning import VisualReasoningEvaluator
from .ocr_document import OCRDocumentEvaluator
from .hallucination import HallucinationEvaluator
from .spatial_awareness import SpatialAwarenessEvaluator

__all__ = [
    "BaseMetricEvaluator",
    "evaluate_general_efficiency",
    "VQAEvaluator",
    "CaptioningEvaluator",
    "VisualReasoningEvaluator",
    "OCRDocumentEvaluator",
    "HallucinationEvaluator",
    "SpatialAwarenessEvaluator",
]
