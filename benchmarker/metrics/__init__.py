"""
Metric group implementations for VLM compression benchmarking.

This package will contain individual modules for each of the six capability
metric groups. Implementation is planned for Phase 2.

Planned modules:
    - general.py          — General efficiency metrics (size, latency, etc.)
    - vqa.py              — Visual Question Answering (VQA-v2)
    - captioning.py       — Image Captioning (COCO Captions, CIDEr)
    - visual_reasoning.py — Visual Reasoning (GQA)
    - ocr_document.py     — OCR / Document Understanding (TextVQA, ANLS)
    - hallucination.py    — Hallucination detection (CHAIR-S, lower-is-better)
    - spatial_awareness.py — Spatial Awareness (VSR)
"""

__all__: list[str] = []
