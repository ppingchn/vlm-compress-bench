"""
Image captioning metric evaluator for VLM compression benchmarking.

This module evaluates image captioning performance using standard metrics:
CIDEr, BLEU-4, METEOR, and ROUGE-L on dataset splits such as COCO Captions.
"""

from __future__ import annotations

import collections
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from benchmarker.metrics.base import BaseMetricEvaluator
from benchmarker.utils import generate_text, normalize_answer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric Library Imports (with Graceful Fallbacks)
# ---------------------------------------------------------------------------

_PYCOCOEVALCAP_AVAILABLE = False
try:
    from pycocoevalcap.cider.cider import Cider
    _PYCOCOEVALCAP_AVAILABLE = True
except ImportError:
    Cider = None
    logger.debug("pycocoevalcap not available; CIDEr fallback will be used if needed.")

_EVALUATE_AVAILABLE = False
try:
    import evaluate
    _EVALUATE_AVAILABLE = True
except ImportError:
    evaluate = None
    logger.debug("evaluate library not available.")

_NLTK_AVAILABLE = False
try:
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    from nltk.translate.meteor_score import meteor_score

    for _pkg in ["punkt", "punkt_tab", "wordnet"]:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass
    _NLTK_AVAILABLE = True
except ImportError:
    nltk = None
    corpus_bleu = None
    SmoothingFunction = None
    meteor_score = None
    logger.debug("NLTK not available; BLEU-4 and METEOR metrics will return None.")

_ROUGE_AVAILABLE = False
try:
    from rouge_score import rouge_scorer
    _ROUGE_AVAILABLE = True
except ImportError:
    rouge_scorer = None
    logger.debug("rouge_score not available; ROUGE-L metric will return None.")

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None
    logger.debug("datasets library not available.")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _extract_captions_and_image(sample: Dict[str, Any]) -> Tuple[Any, List[str]]:
    """Extract PIL image and reference captions list from a dataset sample.

    Supports multiple common HuggingFace dataset formats for COCO Captions.

    Args:
        sample: A single item from the dataset split.

    Returns:
        A tuple of (image, list_of_caption_strings).
    """
    image = sample.get("image")
    if image is None and "img" in sample:
        image = sample["img"]

    captions: List[str] = []

    if "sentences" in sample:
        sentences = sample["sentences"]
        if isinstance(sentences, list):
            for s in sentences:
                if isinstance(s, dict):
                    raw_text = s.get("raw") or s.get("caption") or s.get("sent")
                    if raw_text:
                        captions.append(str(raw_text))
                elif isinstance(s, str):
                    captions.append(s)
        elif isinstance(sentences, dict) and "raw" in sentences:
            raw_list = sentences["raw"]
            if isinstance(raw_list, list):
                captions.extend([str(c) for c in raw_list])

    elif "captions" in sample:
        raw_captions = sample["captions"]
        if isinstance(raw_captions, list):
            for c in raw_captions:
                if isinstance(c, dict):
                    text = c.get("text") or c.get("caption") or c.get("raw")
                    if text:
                        captions.append(str(text))
                elif isinstance(c, str):
                    captions.append(c)
        elif isinstance(raw_captions, str):
            captions.append(raw_captions)

    elif "caption" in sample:
        raw_caption = sample["caption"]
        if isinstance(raw_caption, list):
            for c in raw_caption:
                if isinstance(c, dict):
                    text = c.get("text") or c.get("caption")
                    if text:
                        captions.append(str(text))
                elif isinstance(c, str):
                    captions.append(c)
        elif isinstance(raw_caption, str):
            captions.append(raw_caption)

    return image, captions


def _tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercased tokens.

    Uses NLTK ``word_tokenize`` if available, falling back to basic split.

    Args:
        text: Text string to tokenize.

    Returns:
        List of token strings.
    """
    if _NLTK_AVAILABLE and nltk is not None:
        try:
            return nltk.word_tokenize(text.lower())
        except Exception:
            pass
    return normalize_answer(text).split()


def _compute_cider_fallback(
    predictions: List[str],
    references_list: List[List[str]],
    n: int = 4,
    sigma: float = 6.0,
) -> float:
    """Simplified CIDEr implementation using TF-IDF weighted n-gram matching.

    Fallback used when neither ``pycocoevalcap`` nor ``evaluate`` is present.

    Args:
        predictions: List of model prediction strings.
        references_list: List of ground-truth reference caption lists.
        n: Maximum n-gram length (default: 4).
        sigma: Gaussian penalty bandwidth for length difference (default: 6.0).

    Returns:
        CIDEr score (float >= 0.0).
    """
    if not predictions or not references_list or len(predictions) != len(references_list):
        return 0.0

    def get_ngrams(words: List[str], n_val: int) -> List[Tuple[str, ...]]:
        return [tuple(words[i : i + n_val]) for i in range(len(words) - n_val + 1)]

    preds_tok = [normalize_answer(p).split() for p in predictions]
    refs_tok = [[normalize_answer(r).split() for r in refs] for refs in references_list]

    num_samples = len(predictions)
    sample_cider_scores: List[float] = []

    for sample_idx in range(num_samples):
        cand_words = preds_tok[sample_idx]
        sample_refs_words = refs_tok[sample_idx]
        if not cand_words or not sample_refs_words:
            sample_cider_scores.append(0.0)
            continue

        ngram_scores: List[float] = []
        for n_val in range(1, n + 1):
            cand_ngrams = get_ngrams(cand_words, n_val)
            ref_ngrams_list = [get_ngrams(r, n_val) for r in sample_refs_words]

            all_sentences = ref_ngrams_list + ([cand_ngrams] if cand_ngrams else [])
            doc_freq: Dict[Tuple[str, ...], int] = {}
            for sent in all_sentences:
                seen = set(sent)
                for ng in seen:
                    doc_freq[ng] = doc_freq.get(ng, 0) + 1

            total_docs = len(all_sentences)

            def get_tfidf_vec(ngrams: List[Tuple[str, ...]]) -> Dict[Tuple[str, ...], float]:
                if not ngrams:
                    return {}
                counts = collections.Counter(ngrams)
                vec: Dict[Tuple[str, ...], float] = {}
                for ng, count in counts.items():
                    df = doc_freq.get(ng, 1)
                    idf = math.log(max(1.0, total_docs / max(1.0, df)))
                    tf = count / len(ngrams)
                    vec[ng] = tf * idf
                return vec

            cand_vec = get_tfidf_vec(cand_ngrams)
            cand_norm = math.sqrt(sum(v * v for v in cand_vec.values()))

            ref_sims: List[float] = []
            for ref_ngrams in ref_ngrams_list:
                ref_vec = get_tfidf_vec(ref_ngrams)
                ref_norm = math.sqrt(sum(v * v for v in ref_vec.values()))

                if cand_norm == 0 or ref_norm == 0:
                    sim = 0.0
                else:
                    dot = sum(cand_vec[ng] * ref_vec[ng] for ng in cand_vec if ng in ref_vec)
                    sim = dot / (cand_norm * ref_norm)

                delta_len = len(cand_words) - len(ref_ngrams)
                penalty = math.exp(-(delta_len ** 2) / (2 * (sigma ** 2)))
                ref_sims.append(sim * penalty)

            avg_sim = sum(ref_sims) / len(ref_sims) if ref_sims else 0.0
            ngram_scores.append(avg_sim)

        sample_score = (sum(ngram_scores) / len(ngram_scores)) * 10.0
        sample_cider_scores.append(sample_score)

    return float(sum(sample_cider_scores) / len(sample_cider_scores)) if sample_cider_scores else 0.0


# ---------------------------------------------------------------------------
# Evaluator Class
# ---------------------------------------------------------------------------

class CaptioningEvaluator(BaseMetricEvaluator):
    """Evaluator for image captioning tasks on COCO Captions."""

    @property
    def group_name(self) -> str:
        """Return the metric group key for snapshot schema ('captioning')."""
        return "captioning"

    def evaluate(
        self,
        model: Any,
        processor: Any,
        *,
        device: str = "cuda",
        max_samples: Optional[int] = None,
        batch_size: int = 1,
        max_new_tokens: int = 128,
        dataset_name: str = "HuggingFaceM4/COCO",
        dataset: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Evaluate VLM image captioning generation.

        Args:
            model: HuggingFace-compatible VLM supporting generate().
            processor: Processor/tokenizer for vision + text inputs.
            device: Torch device ('cuda', 'cpu', etc.).
            max_samples: Maximum samples to evaluate (optional).
            batch_size: Batch size for inference (default: 1).
            max_new_tokens: Maximum tokens per generation step.
            dataset_name: Name of HuggingFace dataset if dataset is None.
            dataset: Pre-loaded dataset split (optional).
            **kwargs: Extra parameters.

        Returns:
            Dictionary matching the captioning schema:
            {
                "cider": float,
                "bleu4": float | None,
                "meteor": float | None,
                "rouge_l": float | None,
                "dataset": str,
                "num_samples": int,
            }
        """
        if dataset is None:
            if load_dataset is None:
                logger.error("HuggingFace `datasets` library is not installed.")
                return {
                    "cider": 0.0,
                    "bleu4": None,
                    "meteor": None,
                    "rouge_l": None,
                    "dataset": "COCO Captions" if dataset_name == "HuggingFaceM4/COCO" else str(dataset_name),
                    "num_samples": 0,
                }

            logger.info("Loading dataset '%s'...", dataset_name)
            try:
                ds = load_dataset(dataset_name, split="val")
            except Exception:
                try:
                    ds = load_dataset(dataset_name, split="validation")
                except Exception as err:
                    logger.error("Failed to load dataset '%s': %s", dataset_name, err)
                    return {
                        "cider": 0.0,
                        "bleu4": None,
                        "meteor": None,
                        "rouge_l": None,
                        "dataset": "COCO Captions" if dataset_name == "HuggingFaceM4/COCO" else str(dataset_name),
                        "num_samples": 0,
                    }
        else:
            ds = dataset

        if max_samples is not None and max_samples > 0:
            if hasattr(ds, "select"):
                ds = ds.select(range(min(max_samples, len(ds))))
            else:
                ds = list(ds)[:max_samples]

        self._log_start(len(ds))

        predictions: List[str] = []
        references_list: List[List[str]] = []
        prompt = "Describe this image in a single sentence."

        for sample in tqdm(ds, desc="Evaluating Captioning"):
            try:
                image, captions = _extract_captions_and_image(sample)
                if image is None or not captions:
                    logger.warning("Skipping sample due to missing image or reference captions.")
                    continue

                pred = generate_text(
                    model=model,
                    processor=processor,
                    image=image,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=max_new_tokens,
                )
                predictions.append(pred)
                references_list.append(captions)
            except Exception as exc:
                logger.warning("Error generating caption for sample: %s", exc)
                continue

        dataset_label = "COCO Captions" if dataset_name == "HuggingFaceM4/COCO" else str(dataset_name)

        if not predictions:
            logger.warning("No samples were successfully evaluated.")
            results = {
                "cider": 0.0,
                "bleu4": None,
                "meteor": None,
                "rouge_l": None,
                "dataset": dataset_label,
                "num_samples": 0,
            }
            self._log_done(results)
            return results

        # Compute each metric independently
        cider_score = self._compute_cider(predictions, references_list)
        bleu4_score = self._compute_bleu4(predictions, references_list)
        meteor_score_val = self._compute_meteor(predictions, references_list)
        rouge_l_score = self._compute_rouge_l(predictions, references_list)

        results = {
            "cider": float(cider_score if cider_score is not None else 0.0),
            "bleu4": float(bleu4_score) if bleu4_score is not None else None,
            "meteor": float(meteor_score_val) if meteor_score_val is not None else None,
            "rouge_l": float(rouge_l_score) if rouge_l_score is not None else None,
            "dataset": dataset_label,
            "num_samples": len(predictions),
        }

        self._log_done(results)
        return results

    def _compute_cider(
        self, predictions: List[str], references_list: List[List[str]]
    ) -> float:
        """Compute CIDEr score using pycocoevalcap, evaluate, or fallback."""
        if not predictions or not references_list:
            return 0.0

        if _PYCOCOEVALCAP_AVAILABLE and Cider is not None:
            try:
                gts = {i: refs for i, refs in enumerate(references_list)}
                res = {i: [pred] for i, pred in enumerate(predictions)}
                cider_scorer = Cider()
                score, _ = cider_scorer.compute_score(gts, res)
                return float(score)
            except Exception as err:
                logger.warning("pycocoevalcap CIDEr computation failed: %s", err)

        if _EVALUATE_AVAILABLE and evaluate is not None:
            try:
                cider_eval = evaluate.load("cider")
                res = cider_eval.compute(
                    predictions=predictions, references=references_list
                )
                if "cider" in res:
                    return float(res["cider"])
            except Exception as err:
                logger.warning("HuggingFace evaluate CIDEr computation failed: %s", err)

        logger.info("Using simplified TF-IDF n-gram fallback for CIDEr.")
        return _compute_cider_fallback(predictions, references_list)

    def _compute_bleu4(
        self, predictions: List[str], references_list: List[List[str]]
    ) -> Optional[float]:
        """Compute corpus BLEU-4 score using NLTK."""
        if not predictions or not references_list or not _NLTK_AVAILABLE or corpus_bleu is None:
            if not _NLTK_AVAILABLE:
                logger.warning("NLTK is not available; skipping BLEU-4 computation.")
            return None

        try:
            list_of_references: List[List[List[str]]] = []
            hypotheses: List[List[str]] = []

            for pred, refs in zip(predictions, references_list):
                hyp_tokens = _tokenize_text(pred)
                ref_tokens = [_tokenize_text(r) for r in refs]
                hypotheses.append(hyp_tokens)
                list_of_references.append(ref_tokens)

            sf = SmoothingFunction().method1
            score = corpus_bleu(
                list_of_references, hypotheses, smoothing_function=sf
            )
            return float(score)
        except Exception as err:
            logger.warning("BLEU-4 computation failed: %s", err)
            return None

    def _compute_meteor(
        self, predictions: List[str], references_list: List[List[str]]
    ) -> Optional[float]:
        """Compute average METEOR score across samples using NLTK."""
        if not predictions or not references_list or not _NLTK_AVAILABLE or meteor_score is None:
            if not _NLTK_AVAILABLE:
                logger.warning("NLTK is not available; skipping METEOR computation.")
            return None

        try:
            scores: List[float] = []
            for pred, refs in zip(predictions, references_list):
                hyp_tokens = _tokenize_text(pred)
                ref_tokens = [_tokenize_text(r) for r in refs]
                sample_score = meteor_score(ref_tokens, hyp_tokens)
                scores.append(float(sample_score))

            return float(sum(scores) / len(scores)) if scores else None
        except Exception as err:
            logger.warning("METEOR computation failed: %s", err)
            return None

    def _compute_rouge_l(
        self, predictions: List[str], references_list: List[List[str]]
    ) -> Optional[float]:
        """Compute average max-reference ROUGE-L score using rouge_score."""
        if not predictions or not references_list or not _ROUGE_AVAILABLE or rouge_scorer is None:
            if not _ROUGE_AVAILABLE:
                logger.warning("rouge_score is not available; skipping ROUGE-L computation.")
            return None

        try:
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            scores: List[float] = []

            for pred, refs in zip(predictions, references_list):
                ref_scores = [
                    scorer.score(ref, pred)["rougeL"].fmeasure
                    for ref in refs
                ]
                if ref_scores:
                    scores.append(float(max(ref_scores)))

            return float(sum(scores) / len(scores)) if scores else None
        except Exception as err:
            logger.warning("ROUGE-L computation failed: %s", err)
            return None
