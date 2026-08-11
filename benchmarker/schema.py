"""
JSON schema definition for VLM compression snapshots.

This module provides the schema definition for snapshots, a validation function,
and a factory function to create an empty snapshot template.

A snapshot captures the complete evaluation state of a model (baseline or
compressed) at a specific compression step. The schema follows JSON Schema
Draft 7 and covers: model metadata, general efficiency metrics, six capability
metric groups, evaluation configuration, and user-facing labels.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from jsonschema import validate, ValidationError  # noqa: F401 (re-exported)


# ---------------------------------------------------------------------------
# Schema Definition (JSON Schema Draft 7)
# ---------------------------------------------------------------------------

SNAPSHOT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "VLM Compression Snapshot Schema",
    "description": (
        "A snapshot capturing the state of a VLM at a specific compression "
        "step, including model metadata, efficiency metrics, capability "
        "metrics across six groups, and evaluation configuration."
    ),
    "type": "object",
    "properties": {
        "schema_version": {
            "type": "string",
            "description": "Schema version identifier, e.g. '1.0'.",
        },
        "snapshot_id": {
            "type": "string",
            "description": "Unique UUID for this snapshot.",
        },
        "timestamp": {
            "type": "string",
            "format": "date-time",
            "description": "ISO 8601 datetime when the snapshot was created.",
        },
        "model_info": {
            "type": "object",
            "description": "Metadata describing the model under evaluation.",
            "properties": {
                "model_name": {"type": "string"},
                "model_family": {"type": "string"},
                "base_llm": {"type": "string"},
                "vision_encoder": {"type": "string"},
                "precision": {"type": "string"},
                "compression_method": {"type": ["string", "null"]},
                "compression_config": {"type": ["object", "null"]},
                "is_baseline": {"type": "boolean"},
            },
            "required": [
                "model_name",
                "model_family",
                "base_llm",
                "vision_encoder",
                "precision",
                "compression_method",
                "compression_config",
                "is_baseline",
            ],
        },
        "general_efficiency": {
            "type": "object",
            "description": "Hardware and efficiency metrics.",
            "properties": {
                "model_size_mb": {"type": "number"},
                "parameter_count": {"type": "integer"},
                "compression_ratio": {"type": "number"},
                "inference_latency_ms": {"type": "number"},
                "gpu_memory_mb": {"type": "number"},
                "throughput_tokens_per_sec": {"type": "number"},
            },
            "required": [
                "model_size_mb",
                "parameter_count",
                "compression_ratio",
                "inference_latency_ms",
                "gpu_memory_mb",
                "throughput_tokens_per_sec",
            ],
        },
        "metrics": {
            "type": "object",
            "description": "Capability metrics across six evaluation groups.",
            "properties": {
                "vqa": {
                    "type": "object",
                    "properties": {
                        "accuracy": {"type": "number"},
                        "yes_no_accuracy": {"type": ["number", "null"]},
                        "number_accuracy": {"type": ["number", "null"]},
                        "other_accuracy": {"type": ["number", "null"]},
                        "dataset": {"type": "string"},
                        "num_samples": {"type": "integer"},
                    },
                    "required": [
                        "accuracy", "yes_no_accuracy", "number_accuracy",
                        "other_accuracy", "dataset", "num_samples",
                    ],
                },
                "captioning": {
                    "type": "object",
                    "properties": {
                        "cider": {"type": "number"},
                        "bleu4": {"type": ["number", "null"]},
                        "meteor": {"type": ["number", "null"]},
                        "rouge_l": {"type": ["number", "null"]},
                        "dataset": {"type": "string"},
                        "num_samples": {"type": "integer"},
                    },
                    "required": [
                        "cider", "bleu4", "meteor", "rouge_l",
                        "dataset", "num_samples",
                    ],
                },
                "visual_reasoning": {
                    "type": "object",
                    "properties": {
                        "accuracy": {"type": "number"},
                        "consistency": {"type": ["number", "null"]},
                        "validity": {"type": ["number", "null"]},
                        "plausibility": {"type": ["number", "null"]},
                        "dataset": {"type": "string"},
                        "num_samples": {"type": "integer"},
                    },
                    "required": [
                        "accuracy", "consistency", "validity",
                        "plausibility", "dataset", "num_samples",
                    ],
                },
                "ocr_document": {
                    "type": "object",
                    "properties": {
                        "anls": {"type": "number"},
                        "exact_match_accuracy": {"type": ["number", "null"]},
                        "partial_match_rate": {"type": ["number", "null"]},
                        "dataset": {"type": "string"},
                        "num_samples": {"type": "integer"},
                    },
                    "required": [
                        "anls", "exact_match_accuracy",
                        "partial_match_rate", "dataset", "num_samples",
                    ],
                },
                "hallucination": {
                    "type": "object",
                    "description": (
                        "Hallucination metrics. NOTE: CHAIR-S is "
                        "lower-is-better (unlike all other metrics)."
                    ),
                    "properties": {
                        "chair_s": {"type": "number"},
                        "chair_i": {"type": ["number", "null"]},
                        "coverage": {"type": ["number", "null"]},
                        "dataset": {"type": "string"},
                        "num_samples": {"type": "integer"},
                    },
                    "required": [
                        "chair_s", "chair_i", "coverage",
                        "dataset", "num_samples",
                    ],
                },
                "spatial_awareness": {
                    "type": "object",
                    "properties": {
                        "accuracy": {"type": "number"},
                        "above_below_accuracy": {"type": ["number", "null"]},
                        "left_right_accuracy": {"type": ["number", "null"]},
                        "near_far_accuracy": {"type": ["number", "null"]},
                        "inside_outside_accuracy": {"type": ["number", "null"]},
                        "gap_above_random": {"type": ["number", "null"]},
                        "dataset": {"type": "string"},
                        "num_samples": {"type": "integer"},
                    },
                    "required": [
                        "accuracy", "above_below_accuracy",
                        "left_right_accuracy", "near_far_accuracy",
                        "inside_outside_accuracy", "gap_above_random",
                        "dataset", "num_samples",
                    ],
                },
            },
            "required": [
                "vqa", "captioning", "visual_reasoning",
                "ocr_document", "hallucination", "spatial_awareness",
            ],
        },
        "evaluation_config": {
            "type": "object",
            "description": "Configuration used during evaluation.",
            "properties": {
                "device": {"type": "string"},
                "batch_size": {"type": "integer"},
                "max_new_tokens": {"type": "integer"},
                "seed": {"type": ["integer", "null"]},
            },
            "required": ["device", "batch_size", "max_new_tokens", "seed"],
        },
        "label": {
            "type": "string",
            "description": "Human-readable label, e.g. 'Baseline (FP16)'.",
        },
        "notes": {
            "type": ["string", "null"],
            "description": "Optional freeform notes about this snapshot.",
        },
    },
    "required": [
        "schema_version",
        "snapshot_id",
        "timestamp",
        "model_info",
        "general_efficiency",
        "metrics",
        "evaluation_config",
        "label",
        "notes",
    ],
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_snapshot(data: dict) -> None:
    """Validate a snapshot dictionary against the SNAPSHOT_SCHEMA.

    Args:
        data: The snapshot dictionary to validate.

    Raises:
        jsonschema.ValidationError: If the snapshot does not conform to the
            schema (missing required fields, wrong types, etc.).
    """
    validate(instance=data, schema=SNAPSHOT_SCHEMA)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_empty_snapshot() -> Dict[str, Any]:
    """Create a valid empty snapshot template with default / placeholder values.

    The returned dictionary passes ``validate_snapshot`` out-of-the-box.
    Callers should overwrite placeholder values (model name, metrics, etc.)
    with real data before persisting the snapshot.

    Returns:
        A dictionary representing an empty snapshot conforming to the schema.
    """
    return {
        "schema_version": "1.0",
        "snapshot_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_info": {
            "model_name": "",
            "model_family": "LLaVA-1.5",
            "base_llm": "Vicuna-7B",
            "vision_encoder": "CLIP-ViT-L/14",
            "precision": "fp16",
            "compression_method": None,
            "compression_config": None,
            "is_baseline": True,
        },
        "general_efficiency": {
            "model_size_mb": 0.0,
            "parameter_count": 0,
            "compression_ratio": 1.0,
            "inference_latency_ms": 0.0,
            "gpu_memory_mb": 0.0,
            "throughput_tokens_per_sec": 0.0,
        },
        "metrics": {
            "vqa": {
                "accuracy": 0.0,
                "yes_no_accuracy": None,
                "number_accuracy": None,
                "other_accuracy": None,
                "dataset": "VQA-v2",
                "num_samples": 0,
            },
            "captioning": {
                "cider": 0.0,
                "bleu4": None,
                "meteor": None,
                "rouge_l": None,
                "dataset": "COCO Captions",
                "num_samples": 0,
            },
            "visual_reasoning": {
                "accuracy": 0.0,
                "consistency": None,
                "validity": None,
                "plausibility": None,
                "dataset": "GQA",
                "num_samples": 0,
            },
            "ocr_document": {
                "anls": 0.0,
                "exact_match_accuracy": None,
                "partial_match_rate": None,
                "dataset": "TextVQA",
                "num_samples": 0,
            },
            "hallucination": {
                "chair_s": 0.0,
                "chair_i": None,
                "coverage": None,
                "dataset": "COCO",
                "num_samples": 0,
            },
            "spatial_awareness": {
                "accuracy": 0.0,
                "above_below_accuracy": None,
                "left_right_accuracy": None,
                "near_far_accuracy": None,
                "inside_outside_accuracy": None,
                "gap_above_random": None,
                "dataset": "VSR",
                "num_samples": 0,
            },
        },
        "evaluation_config": {
            "device": "cuda",
            "batch_size": 1,
            "max_new_tokens": 128,
            "seed": None,
        },
        "label": "",
        "notes": None,
    }
