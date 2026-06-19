# VLM Compression Benchmarker

A user-friendly benchmarking toolkit for evaluating the impact of compression on Visual Language Models (VLMs). Bring your baseline and compressed models, run the benchmarker, and explore the results through an interactive comparison dashboard.

Inspired by [LVLM-Compress-Bench](https://arxiv.org/abs/2503.04982).

---

## Overview

Model compression techniques such as quantization, pruning, and knowledge distillation are essential for deploying VLMs in resource-constrained environments. However, compression does not affect all capabilities equally — a model may retain strong VQA performance while significantly degrading in hallucination resistance or spatial reasoning.

This toolkit helps you:
- Evaluate your baseline and compressed models across multiple capability groups
- Capture step-by-step snapshots throughout the compression process
- Visualize capability retention through an interactive Plotly Dash dashboard
- Save and share snapshot results in a portable JSON format

---

## User Journey
1. Your baseline model + compressed model
2. Run snapshot() on each model
3. Benchmarker evaluates across capability groups
4. launch_dashboard(scapshots) opens interactive comparison

---

## Metric Groups

| Group | Primary Metric | Dataset |
|---|---|---|
| General Efficiency | Size, Latency, Compression Ratio | — |
| Visual Question Answering | VQA Accuracy | VQA-v2 |
| Image Captioning | CIDEr | COCO Captions |
| Visual Reasoning | Accuracy | GQA |
| OCR / Document Understanding | ANLS | TextVQA |
| Hallucination | CHAIR-S (lower is better) | COCO |
| Spatial Awareness | Accuracy | VSR |

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run benchmarking

```python
from benchmarker import snapshot, launch_dashboard

baseline_snap = snapshot(baseline_model, label="Baseline (FP16)")
compressed_snap = snapshot(compressed_model, label="W4 Quantized")

launch_dashboard([baseline_snap, compressed_snap])
```

### 3. Save and load snapshots

```python
from benchmarker import save_snapshot, load_snapshot

save_snapshot(baseline_snap, "snapshots/baseline.json")
loaded = load_snapshot("snapshots/baseline.json")
```

---

## Baseline Model

This project uses **LLaVA-1.5-7B** as the reference baseline model.

- Architecture: LLaVA-1.5
- Base LLM: Vicuna-7B
- Vision Encoder: CLIP-ViT-L/14
- Format: FP16 (uncompressed)

---

## Project Structure
vlm-compress-bench/

│

├── README.md
├── requirements.txt
│
├── benchmarker/
│   ├── init.py
│   ├── snapshot.py
│   ├── evaluator.py
│   ├── utils.py
│   └── metrics/
│       ├── general.py
│       ├── vqa.py
│       ├── captioning.py
│       ├── visual_reasoning.py
│       ├── ocr_document.py
│       ├── hallucination.py
│       └── spatial_awareness.py
│
├── dashboard/
│   └── app.py
│
├── notebooks/
│   └── demo_benchmark.ipynb
│
├── snapshots/
│   └── baselines.json
│
└── docs/
└── metrics.md---

## Requirements

- Python 3.9+
- PyTorch
- Transformers (HuggingFace)
- Datasets (HuggingFace)
- Plotly
- Dash
- Pillow
- evaluate

See `requirements.txt` for full version details.

---

## Acknowledgements

This project is developed as a graduation project (專題) and is inspired by the LVLM-Compress-Bench framework published at NAACL 2025.