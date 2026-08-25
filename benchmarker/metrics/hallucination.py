"""
Object hallucination evaluator using CHAIR on COCO.

This module evaluates Visual Language Model (VLM) object hallucination using
CHAIR (Caption Hallucination Assessment with Image Relevance) metrics on the
COCO dataset.

Metrics computed:
    * **CHAIR-S** (Sentence-level): Fraction of generated captions containing at
      least one hallucinated object. NOTE: **LOWER IS BETTER**!
    * **CHAIR-I** (Instance-level): Fraction of all mentioned object instances
      that are hallucinated. NOTE: **LOWER IS BETTER**!
    * **Coverage**: Fraction of ground-truth COCO objects mentioned in generated
      captions (higher is better; prevents gaming empty/sparse captions).

Schema output:
    .. code-block:: python

        {
            "chair_s": float,      # Lower is better!
            "chair_i": float | None,
            "coverage": float | None,
            "dataset": str,
            "num_samples": int,
        }
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image
from tqdm import tqdm

from benchmarker.metrics.base import BaseMetricEvaluator
from benchmarker.utils import generate_text

logger = logging.getLogger(__name__)

#: Mapping of 80 COCO object category names to sets of common English synonyms.
COCO_SYNONYMS: dict[str, set[str]] = {
    "person": {
        "person", "persons", "people", "man", "men", "woman", "women",
        "boy", "boys", "girl", "girls", "child", "children", "kid", "kids",
        "baby", "babies", "toddler", "toddlers", "adult", "adults",
        "lady", "ladies", "guy", "guys", "human", "humans", "pedestrian",
        "pedestrians", "player", "players", "infant", "infants",
    },
    "bicycle": {
        "bicycle", "bicycles", "bike", "bikes", "cycle", "cycles",
        "biker", "bikers", "cyclist", "cyclists",
    },
    "car": {
        "car", "cars", "automobile", "automobiles", "vehicle", "vehicles",
        "auto", "autos", "sedan", "sedans", "suv", "suvs", "taxicab",
        "taxi", "taxis", "cab", "cabs",
    },
    "motorcycle": {
        "motorcycle", "motorcycles", "motorbike", "motorbikes", "scooter",
        "scooters", "moped", "mopeds", "chopper", "choppers",
    },
    "airplane": {
        "airplane", "airplanes", "plane", "planes", "aircraft", "jet",
        "jets", "airliner", "airliners", "biplane", "monoplane",
    },
    "bus": {
        "bus", "buses", "minibus", "minibuses", "schoolbus", "schoolbuses",
    },
    "train": {
        "train", "trains", "locomotive", "locomotives", "subway", "subways",
        "boxcar", "boxcars",
    },
    "truck": {
        "truck", "trucks", "pickup", "pickups", "trailer", "trailers",
        "lorry", "lorries",
    },
    "boat": {
        "boat", "boats", "vessel", "vessels", "ship", "ships", "sailboat",
        "sailboats", "yacht", "yachts", "canoe", "canoes", "kayak", "kayaks",
        "ferry", "ferries", "raft", "rafts", "barge", "barges",
    },
    "traffic light": {
        "traffic light", "traffic lights", "traffic signal", "traffic signals",
        "stoplight", "stoplights", "signal light", "signal lights",
    },
    "fire hydrant": {
        "fire hydrant", "fire hydrants", "hydrant", "hydrants",
    },
    "stop sign": {
        "stop sign", "stop signs",
    },
    "parking meter": {
        "parking meter", "parking meters",
    },
    "bench": {
        "bench", "benches", "park bench", "park benches",
    },
    "bird": {
        "bird", "birds", "seagull", "seagulls", "duck", "ducks", "duckling",
        "ducklings", "goose", "geese", "owl", "owls", "pigeon", "pigeons",
        "parrot", "parrots", "eagle", "eagles", "swan", "swans", "crow",
        "crows", "hawk", "hawks", "sparrow", "sparrows", "fowl",
    },
    "cat": {
        "cat", "cats", "kitten", "kittens", "kitty", "feline", "felines",
    },
    "dog": {
        "dog", "dogs", "puppy", "puppies", "doggy", "canine", "canines",
        "hound", "hounds", "pooch", "pooches",
    },
    "horse": {
        "horse", "horses", "pony", "ponies", "stallion", "stallions",
        "mare", "mares", "colt", "colts", "equine",
    },
    "sheep": {
        "sheep", "lamb", "lambs", "ram", "rams", "ewe", "ewes",
    },
    "cow": {
        "cow", "cows", "cattle", "bull", "bulls", "calf", "calves",
        "heifer", "heifers", "ox", "oxen",
    },
    "elephant": {
        "elephant", "elephants",
    },
    "bear": {
        "bear", "bears", "grizzly", "grizzlies",
    },
    "zebra": {
        "zebra", "zebras",
    },
    "giraffe": {
        "giraffe", "giraffes",
    },
    "backpack": {
        "backpack", "backpacks", "knapsack", "knapsacks", "rucksack",
        "rucksacks", "pack", "packs",
    },
    "umbrella": {
        "umbrella", "umbrellas", "parasol", "parasols",
    },
    "handbag": {
        "handbag", "handbags", "purse", "purses", "pocketbook",
        "pocketbooks", "tote", "totes",
    },
    "tie": {
        "tie", "ties", "necktie", "neckties",
    },
    "suitcase": {
        "suitcase", "suitcases", "luggage", "baggage", "suit case",
        "suit cases",
    },
    "frisbee": {
        "frisbee", "frisbees",
    },
    "skis": {
        "ski", "skis", "skiing",
    },
    "snowboard": {
        "snowboard", "snowboards", "snowboarding",
    },
    "sports ball": {
        "sports ball", "sports balls", "ball", "balls", "basketball",
        "basketballs", "soccer ball", "soccer balls", "baseball",
        "baseballs", "football", "footballs", "volleyball", "volleyballs",
        "tennis ball", "tennis balls",
    },
    "kite": {
        "kite", "kites",
    },
    "baseball bat": {
        "baseball bat", "baseball bats", "bat", "bats",
    },
    "baseball glove": {
        "baseball glove", "baseball gloves", "baseball mitt",
        "baseball mitts", "glove", "gloves", "mitt", "mitts",
    },
    "skateboard": {
        "skateboard", "skateboards", "skateboarding",
    },
    "surfboard": {
        "surfboard", "surfboards", "surfboarder", "surfboarders",
    },
    "tennis racket": {
        "tennis racket", "tennis rackets", "tennis racquet",
        "tennis racquets", "racket", "rackets", "racquet", "racquets",
    },
    "bottle": {
        "bottle", "bottles", "flask", "flasks",
    },
    "wine glass": {
        "wine glass", "wine glasses", "wineglass", "wineglasses",
        "goblet", "goblets",
    },
    "cup": {
        "cup", "cups", "mug", "mugs", "teacup", "teacups",
    },
    "fork": {
        "fork", "forks",
    },
    "knife": {
        "knife", "knives",
    },
    "spoon": {
        "spoon", "spoons",
    },
    "bowl": {
        "bowl", "bowls",
    },
    "banana": {
        "banana", "bananas",
    },
    "apple": {
        "apple", "apples",
    },
    "sandwich": {
        "sandwich", "sandwiches", "burger", "burgers", "hamburger",
        "hamburgers",
    },
    "orange": {
        "orange", "oranges",
    },
    "broccoli": {
        "broccoli",
    },
    "carrot": {
        "carrot", "carrots",
    },
    "hot dog": {
        "hot dog", "hot dogs", "hotdog", "hotdogs", "frankfurter",
        "frankfurters",
    },
    "pizza": {
        "pizza", "pizzas",
    },
    "donut": {
        "donut", "donuts", "doughnut", "doughnuts",
    },
    "cake": {
        "cake", "cakes", "cupcake", "cupcakes",
    },
    "chair": {
        "chair", "chairs", "seat", "seats", "armchair", "armchairs",
        "stool", "stools", "recliner", "recliners",
    },
    "couch": {
        "couch", "couches", "sofa", "sofas", "loveseat", "loveseats",
    },
    "potted plant": {
        "potted plant", "potted plants", "pottedplant", "pottedplants",
        "plant", "plants", "houseplant", "houseplants",
    },
    "bed": {
        "bed", "beds", "mattress", "mattresses",
    },
    "dining table": {
        "dining table", "dining tables", "diningtable", "table", "tables",
        "desk", "desks",
    },
    "toilet": {
        "toilet", "toilets", "commode", "commodes",
    },
    "tv": {
        "tv", "tvs", "television", "televisions", "monitor", "monitors",
        "screen", "screens",
    },
    "laptop": {
        "laptop", "laptops", "notebook", "notebooks",
    },
    "mouse": {
        "mouse", "mice", "computer mouse",
    },
    "remote": {
        "remote", "remotes", "remote control", "remote controls",
    },
    "keyboard": {
        "keyboard", "keyboards",
    },
    "cell phone": {
        "cell phone", "cell phones", "cellphone", "cellphones",
        "mobile phone", "mobile phones", "phone", "phones", "smartphone",
        "smartphones",
    },
    "microwave": {
        "microwave", "microwaves", "microwave oven", "microwave ovens",
    },
    "oven": {
        "oven", "ovens", "stove", "stoves",
    },
    "toaster": {
        "toaster", "toasters",
    },
    "sink": {
        "sink", "sinks",
    },
    "refrigerator": {
        "refrigerator", "refrigerators", "fridge", "fridges", "freezer",
        "freezers",
    },
    "book": {
        "book", "books", "textbook", "textbooks",
    },
    "clock": {
        "clock", "clocks",
    },
    "vase": {
        "vase", "vases",
    },
    "scissors": {
        "scissors", "scissor",
    },
    "teddy bear": {
        "teddy bear", "teddy bears", "teddy", "teddies",
    },
    "hair drier": {
        "hair drier", "hair driers", "hairdryer", "hairdryers",
        "hair dryer", "hair dryers",
    },
    "toothbrush": {
        "toothbrush", "toothbrushes",
    },
}


def extract_object_mentions_from_caption(
    caption: str,
    synonym_map: dict[str, set[str]] | None = None,
) -> list[str]:
    """Extract an ordered list of mentioned COCO category names from a caption.

    Scans the caption for synonyms of COCO categories and returns the
    canonical category name for each non-overlapping mention in order of
    appearance in the caption.

    Args:
        caption: Free-form text caption string.
        synonym_map: Dictionary mapping category names to synonym sets.
            Defaults to ``COCO_SYNONYMS``.

    Returns:
        List of canonical COCO category names for each detected object mention.
    """
    if synonym_map is None:
        synonym_map = COCO_SYNONYMS

    if not caption or not caption.strip():
        return []

    cleaned_caption = re.sub(r"[^\w\s]", " ", caption.lower())

    candidates: list[tuple[int, int, str]] = []
    for cat_name, synonyms in synonym_map.items():
        for syn in synonyms:
            syn_clean = syn.lower().strip()
            if not syn_clean:
                continue
            pattern = r"\b" + re.escape(syn_clean) + r"\b"
            for match in re.finditer(pattern, cleaned_caption):
                candidates.append((match.start(), match.end(), cat_name))

    if not candidates:
        return []

    # Sort candidates by length (end - start) descending, then start index ascending
    candidates.sort(key=lambda x: (-(x[1] - x[0]), x[0]))

    selected_matches: list[tuple[int, int, str]] = []
    occupied_mask = [False] * len(cleaned_caption)

    for start, end, cat in candidates:
        if not any(occupied_mask[start:end]):
            selected_matches.append((start, end, cat))
            for i in range(start, end):
                occupied_mask[i] = True

    # Re-sort selected matches by start position in text to preserve order
    selected_matches.sort(key=lambda x: x[0])

    return [cat for _, _, cat in selected_matches]


def extract_objects_from_caption(
    caption: str,
    synonym_map: dict[str, set[str]] | None = None,
) -> tuple[set[str], int]:
    """Parse a caption to find mentioned COCO objects.

    Args:
        caption: Free-form text caption.
        synonym_map: Dictionary mapping category names to sets of synonym strings.
            Defaults to ``COCO_SYNONYMS``.

    Returns:
        A tuple of ``(matched_categories, total_mentions)`` where
        ``matched_categories`` is a set of matched COCO category names and
        ``total_mentions`` is the total count of object mentions.
    """
    mentions = extract_object_mentions_from_caption(caption, synonym_map)
    return set(mentions), len(mentions)


def compute_chair(
    generated_captions: list[str],
    ground_truth_objects: list[set[str]],
    synonym_map: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Compute CHAIR-S, CHAIR-I, and coverage metrics for generated captions.

    CHAIR (Caption Hallucination Assessment with Image Relevance):
    - **CHAIR-S**: Sentence-level hallucination rate. Fraction of generated
      captions containing at least one hallucinated object. (LOWER IS BETTER!)
    - **CHAIR-I**: Instance-level hallucination rate. Fraction of all mentioned
      object instances that are hallucinated. (LOWER IS BETTER!)
    - **Coverage**: Ground-truth object recall rate. Fraction of ground-truth
      COCO objects mentioned in generated captions. (HIGHER IS BETTER!)

    Args:
        generated_captions: List of model-generated captions.
        ground_truth_objects: List of sets of ground-truth COCO category names
            for each corresponding image.
        synonym_map: Dictionary mapping category names to synonym sets.
            Defaults to ``COCO_SYNONYMS``.

    Returns:
        A dictionary containing:
        - ``chair_s`` (float): Sentence-level hallucination rate.
        - ``chair_i`` (float | None): Instance-level hallucination rate.
        - ``coverage`` (float | None): GT object coverage rate.
    """
    if synonym_map is None:
        synonym_map = COCO_SYNONYMS

    num_sentences = len(generated_captions)
    if num_sentences == 0:
        return {
            "chair_s": 0.0,
            "chair_i": None,
            "coverage": None,
        }

    hallucinated_sentences = 0
    total_mentions = 0
    hallucinated_mentions = 0
    total_gt_objects = 0
    total_gt_mentioned = 0

    for caption, gt_set in zip(generated_captions, ground_truth_objects):
        mentions = extract_object_mentions_from_caption(caption, synonym_map)
        matched_cats = set(mentions)

        # Normalize ground truth category names against synonym map
        norm_gt_set: set[str] = set()
        for cat in gt_set:
            cat_str = str(cat).strip().lower()
            if cat_str in synonym_map:
                norm_gt_set.add(cat_str)
            else:
                extracted, _ = extract_objects_from_caption(cat_str, synonym_map)
                if extracted:
                    norm_gt_set.update(extracted)
                else:
                    norm_gt_set.add(cat_str)

        # Sentence-level hallucination check
        hallucinated_cats = matched_cats - norm_gt_set
        if len(hallucinated_cats) > 0:
            hallucinated_sentences += 1

        # Instance-level mentions count
        for cat in mentions:
            total_mentions += 1
            if cat not in norm_gt_set:
                hallucinated_mentions += 1

        # Coverage calculation
        total_gt_objects += len(norm_gt_set)
        total_gt_mentioned += len(matched_cats & norm_gt_set)

    chair_s = hallucinated_sentences / num_sentences
    chair_i = (
        (hallucinated_mentions / total_mentions)
        if total_mentions > 0
        else None
    )
    coverage = (
        (total_gt_mentioned / total_gt_objects)
        if total_gt_objects > 0
        else None
    )

    return {
        "chair_s": float(chair_s),
        "chair_i": float(chair_i) if chair_i is not None else None,
        "coverage": float(coverage) if coverage is not None else None,
    }


class HallucinationEvaluator(BaseMetricEvaluator):
    """Evaluator for object hallucination on COCO using CHAIR metrics.

    Subclasses :class:`benchmarker.metrics.base.BaseMetricEvaluator` to measure
    object hallucination in VLM generated captions.

    ⚠️ Note: ``chair_s`` and ``chair_i`` are **LOWER-IS-BETTER** metrics.
    """

    @property
    def group_name(self) -> str:
        """Return the schema group key for hallucination metrics."""
        return "hallucination"

    @staticmethod
    def _extract_image(sample: Any) -> Optional[Image.Image]:
        """Extract a PIL Image in RGB mode from a dataset sample.

        Args:
            sample: Dataset sample item.

        Returns:
            PIL Image object in RGB mode, or None if extraction fails.
        """
        image_val = None
        if isinstance(sample, dict):
            image_val = sample.get("image") or sample.get("img")
        elif hasattr(sample, "image"):
            image_val = getattr(sample, "image")

        if image_val is None:
            return None

        if isinstance(image_val, Image.Image):
            return image_val.convert("RGB")

        if isinstance(image_val, str):
            try:
                return Image.open(image_val).convert("RGB")
            except Exception:
                return None

        if isinstance(image_val, dict) and "bytes" in image_val:
            try:
                return Image.open(io.BytesIO(image_val["bytes"])).convert("RGB")
            except Exception:
                return None

        return None

    @staticmethod
    def _extract_gt_objects(
        sample: Any,
        gt_map_by_image_id: dict[int, set[str]],
        gt_map_by_filename: dict[str, set[str]],
    ) -> set[str]:
        """Extract ground-truth COCO object categories for a dataset sample.

        Args:
            sample: Dataset item.
            gt_map_by_image_id: Mapping of COCO image_id to GT object category sets.
            gt_map_by_filename: Mapping of file_name to GT object category sets.

        Returns:
            Set of ground-truth category names.
        """
        if not isinstance(sample, dict):
            return set()

        # 1. Local annotation file map lookup
        img_id = sample.get("image_id") or sample.get("id")
        if img_id is not None and img_id in gt_map_by_image_id:
            return set(gt_map_by_image_id[img_id])

        file_name = sample.get("file_name") or sample.get("filename")
        if file_name and file_name in gt_map_by_filename:
            return set(gt_map_by_filename[file_name])

        # 2. Direct GT fields on sample
        for field in ("ground_truth_objects", "coco_objects", "gt_objects"):
            val = sample.get(field)
            if isinstance(val, (set, list)):
                return {str(x) for x in val}

        # 3. Object detection field "objects"
        objs = sample.get("objects")
        if isinstance(objs, dict):
            names = objs.get("name") or objs.get("names") or objs.get("category_name")
            if names and isinstance(names, list):
                return {str(x) for x in names}
            categories = objs.get("category") or objs.get("label")
            if categories and isinstance(categories, list):
                str_cats = [str(x) for x in categories if isinstance(x, str)]
                if str_cats:
                    return set(str_cats)
        elif isinstance(objs, list):
            cat_names = set()
            for obj in objs:
                if isinstance(obj, dict):
                    name = (
                        obj.get("name")
                        or obj.get("category_name")
                        or obj.get("category")
                    )
                    if name:
                        cat_names.add(str(name))
            if cat_names:
                return cat_names

        # 4. Extract from human reference captions if present
        ref_captions: list[str] = []
        for field in ("sentences", "captions", "caption", "reference_captions", "references"):
            val = sample.get(field)
            if isinstance(val, str):
                ref_captions.append(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        ref_captions.append(item)
                    elif isinstance(item, dict) and "raw" in item:
                        ref_captions.append(item["raw"])
                    elif isinstance(item, dict) and "caption" in item:
                        ref_captions.append(item["caption"])

        if ref_captions:
            gt_cats: set[str] = set()
            for ref in ref_captions:
                matched, _ = extract_objects_from_caption(ref, COCO_SYNONYMS)
                gt_cats.update(matched)
            return gt_cats

        return set()

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
    ) -> dict[str, Any]:
        """Run object hallucination evaluation using CHAIR on COCO.

        Args:
            model: HuggingFace-compatible VLM with ``model.generate()``.
            processor: Tokenizer / processor handling image + text inputs.
            device: Torch device string (``"cuda"``, ``"cpu"``, etc.).
            max_samples: Optional maximum number of samples to evaluate.
            batch_size: Batch size for evaluation (default: 1).
            max_new_tokens: Maximum new tokens generated per caption.
            **kwargs: Additional evaluation configuration:
                * ``dataset_name`` (str): HuggingFace dataset name (default: ``"HuggingFaceM4/COCO"``).
                * ``dataset`` (Any): Pre-loaded dataset object.
                * ``coco_annotation_file`` (str): Path to local COCO annotation file JSON.
                * ``prompt`` (str): Text prompt (default: ``"Describe this image in detail."``).
                * ``split`` (str): Dataset split (default: ``"validation"``).

        Returns:
            A dictionary conforming to the hallucination metric snapshot schema:

            .. code-block:: python

                {
                    "chair_s": float,
                    "chair_i": float | None,
                    "coverage": float | None,
                    "dataset": str,
                    "num_samples": int,
                }
        """
        dataset_name = str(kwargs.get("dataset_name", "HuggingFaceM4/COCO"))
        dataset = kwargs.get("dataset")
        coco_annotation_file = kwargs.get("coco_annotation_file")
        prompt = str(kwargs.get("prompt", "Describe this image in detail."))
        split = str(kwargs.get("split", "validation"))

        gt_map_by_image_id: dict[int, set[str]] = {}
        gt_map_by_filename: dict[str, set[str]] = {}

        if coco_annotation_file and os.path.exists(coco_annotation_file):
            try:
                with open(coco_annotation_file, "r", encoding="utf-8") as f:
                    coco_data = json.load(f)
                cat_id_to_name = {
                    cat["id"]: cat["name"]
                    for cat in coco_data.get("categories", [])
                }
                img_id_to_file = {
                    img["id"]: img.get("file_name", "")
                    for img in coco_data.get("images", [])
                }
                for ann in coco_data.get("annotations", []):
                    img_id = ann.get("image_id")
                    cat_id = ann.get("category_id")
                    cat_name = cat_id_to_name.get(cat_id)
                    if img_id is not None and cat_name:
                        gt_map_by_image_id.setdefault(img_id, set()).add(cat_name)
                        fname = img_id_to_file.get(img_id)
                        if fname:
                            gt_map_by_filename.setdefault(fname, set()).add(cat_name)
                logger.info(
                    "Loaded %d image annotations from local COCO file %s",
                    len(gt_map_by_image_id),
                    coco_annotation_file,
                )
            except Exception as e:
                logger.warning(
                    "Failed to parse local COCO annotation file %s: %s",
                    coco_annotation_file,
                    e,
                )

        if dataset is None:
            logger.info("Loading dataset '%s' (split='%s')…", dataset_name, split)
            try:
                from datasets import load_dataset
                dataset = load_dataset(dataset_name, split=split)
            except Exception as e:
                logger.error("Failed to load dataset '%s': %s", dataset_name, e)
                raise RuntimeError(f"Could not load dataset {dataset_name}: {e}") from e

        if max_samples is not None and max_samples > 0:
            if hasattr(dataset, "select"):
                dataset = dataset.select(range(min(len(dataset), max_samples)))
            else:
                dataset = dataset[:max_samples]

        num_samples = len(dataset)
        self._log_start(num_samples)

        generated_captions: list[str] = []
        ground_truth_objects: list[set[str]] = []

        for idx, sample in enumerate(tqdm(dataset, desc="Evaluating CHAIR Hallucination")):
            try:
                image = self._extract_image(sample)
                if image is None:
                    logger.warning("Sample %d has no valid image, skipping.", idx)
                    continue

                gt_objs = self._extract_gt_objects(
                    sample,
                    gt_map_by_image_id,
                    gt_map_by_filename,
                )

                caption = generate_text(
                    model,
                    processor,
                    image=image,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=max_new_tokens,
                )

                generated_captions.append(caption)
                ground_truth_objects.append(gt_objs)

            except Exception as e:
                logger.warning("Failed to evaluate sample %d: %s", idx, e)
                continue

        results = compute_chair(generated_captions, ground_truth_objects, COCO_SYNONYMS)

        metrics: dict[str, Any] = {
            "chair_s": results["chair_s"],
            "chair_i": results["chair_i"],
            "coverage": results["coverage"],
            "dataset": dataset_name,
            "num_samples": len(generated_captions),
        }

        self._log_done(metrics)
        return metrics
