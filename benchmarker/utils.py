"""
Shared utilities for VLM compression benchmarking.

This module provides common text-processing, answer-extraction, and
inference helpers used by the metric evaluators.  Centralising them here
avoids duplicated normalisation logic across metric modules.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Normalize a free-form answer string for comparison.

    The following transformations are applied in order:

    1. Strip leading/trailing whitespace.
    2. Convert to lowercase.
    3. Remove punctuation (keep alphanumeric and spaces).
    4. Remove articles (a, an, the).
    5. Collapse multiple spaces into one.
    6. Strip again.

    This matches the normalization used by VQA-v2 official evaluation and
    is broadly reusable across other benchmarks.

    Args:
        text: Raw model output or ground-truth answer.

    Returns:
        A cleaned, lowercased answer string.
    """
    text = text.strip().lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

#: Patterns that indicate a positive / affirmative answer.
_YES_PATTERNS: set[str] = {"yes", "true", "correct", "right"}

#: Patterns that indicate a negative / rejecting answer.
_NO_PATTERNS: set[str] = {"no", "false", "incorrect", "wrong"}


def extract_yes_no(text: str) -> Optional[str]:
    """Extract a yes/no signal from free-form model output.

    Looks at the first meaningful word of the response.  Returns
    ``"yes"``, ``"no"``, or ``None`` if the signal is ambiguous.

    Args:
        text: Raw model output.

    Returns:
        ``"yes"``, ``"no"``, or ``None``.
    """
    cleaned = normalize_answer(text)
    if not cleaned:
        return None

    first_word = cleaned.split()[0]
    if first_word in _YES_PATTERNS:
        return "yes"
    if first_word in _NO_PATTERNS:
        return "no"
    return None


def extract_true_false(text: str) -> Optional[bool]:
    """Extract a True/False signal from free-form model output.

    Interprets affirmative words as ``True`` and negative words as
    ``False``.  Returns ``None`` if the output is ambiguous.

    Args:
        text: Raw model output.

    Returns:
        ``True``, ``False``, or ``None``.
    """
    signal = extract_yes_no(text)
    if signal == "yes":
        return True
    if signal == "no":
        return False
    return None


def extract_first_word(text: str) -> str:
    """Return the first whitespace-delimited token, lowercased.

    Args:
        text: Raw model output.

    Returns:
        The first word, or ``""`` if *text* is empty.
    """
    cleaned = text.strip().lower()
    if not cleaned:
        return ""
    return cleaned.split()[0]


def extract_short_answer(text: str, max_words: int = 5) -> str:
    """Extract a short answer from the beginning of model output.

    Takes the first *max_words* words after normalisation.  Useful for
    benchmarks where the expected answer is a brief phrase.

    Args:
        text: Raw model output.
        max_words: Maximum number of words to keep.

    Returns:
        A short, normalised answer string.
    """
    cleaned = normalize_answer(text)
    words = cleaned.split()
    return " ".join(words[:max_words])


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def generate_text(
    model: Any,
    processor: Any,
    *,
    image: Any,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 128,
) -> str:
    """Run a single model.generate() call and decode the output.

    This is a convenience wrapper that handles the common pattern of:
    1. Process image + prompt into model inputs.
    2. Move inputs to the correct device.
    3. Generate output tokens.
    4. Decode only the *new* tokens (strip the prompt).

    Args:
        model: A HuggingFace VLM with a ``generate()`` method.
        processor: The paired processor (handles image + text).
        image: A PIL Image or similar input.
        prompt: The text prompt to send to the model.
        device: Torch device string.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        The decoded model output as a string (prompt tokens excluded).
    """
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # Decode only the generated (new) tokens
    generated_ids = output_ids[0, input_len:]
    text = processor.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


# ---------------------------------------------------------------------------
# Levenshtein / ANLS helper
# ---------------------------------------------------------------------------

def normalized_levenshtein_distance(s1: str, s2: str) -> float:
    """Compute the normalised Levenshtein distance between two strings.

    Returns a value in [0, 1] where 0 means identical strings and 1
    means completely different.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        The normalised edit distance.
    """
    if s1 == s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 1.0

    # Standard DP Levenshtein
    matrix = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        prev = matrix[0]
        matrix[0] = i
        for j in range(1, len2 + 1):
            temp = matrix[j]
            if s1[i - 1] == s2[j - 1]:
                matrix[j] = prev
            else:
                matrix[j] = 1 + min(prev, matrix[j], matrix[j - 1])
            prev = temp

    return matrix[len2] / max(len1, len2)


def anls_score(
    prediction: str,
    ground_truths: List[str],
    threshold: float = 0.5,
) -> float:
    """Compute the ANLS (Average Normalised Levenshtein Similarity) score.

    For each ground-truth answer, compute ``1 - NLD(pred, gt)`` if
    ``NLD < threshold``, else 0.  Return the **maximum** score across
    all ground truths.

    Args:
        prediction: The model's predicted answer (normalised).
        ground_truths: List of acceptable ground-truth answers.
        threshold: NLD threshold below which the answer is considered
            partially correct.

    Returns:
        ANLS score in [0, 1].
    """
    best = 0.0
    for gt in ground_truths:
        nld = normalized_levenshtein_distance(prediction, gt)
        if nld < threshold:
            best = max(best, 1.0 - nld)
    return best
