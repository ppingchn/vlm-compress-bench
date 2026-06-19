# Metric Reference

This document describes each metric group used in the VLM Compression Benchmarker, including the primary metric, supporting metrics, evaluation dataset, and important notes for interpretation.

---

## General Efficiency Metrics

These metrics are always captured regardless of which capability groups are evaluated. They describe the compression outcome in terms of model size and runtime performance.

| Metric | Description |
|---|---|
| Model Size (MB) | Total size of model weights on disk |
| Compression Ratio | Baseline model size divided by compressed model size |
| Parameter Count | Total number of model parameters |
| Inference Latency (ms) | Average time to generate a response per sample |
| GPU Memory (MB) | Peak VRAM usage during inference |
| Throughput (tokens/sec) | Number of tokens generated per second |

---

## Visual Question Answering (VQA)

**Dataset:** VQA-v2  
**Primary Metric:** VQA Accuracy  

### Description
VQA Accuracy uses soft scoring based on agreement across multiple human annotators rather than strict exact match. A generated answer scores full marks if at least three annotators provided the same answer, and partial marks if fewer annotators agreed.

### Why This Metric
VQA-v2 is the most established benchmark for general visual question answering. It covers a wide range of image types and question formats, making it a reliable indicator of overall VLM capability under compression.

### Supporting Metrics
| Metric | Description |
|---|---|
| Yes/No Accuracy | Accuracy on binary answer questions — most robust under compression |
| Number Accuracy | Accuracy on numeric answer questions — degrades faster under compression |
| Other Accuracy | Accuracy on open-ended questions — most sensitive to compression |

---

## Image Captioning

**Dataset:** COCO Captions  
**Primary Metric:** CIDEr  

### Description
CIDEr (Consensus-based Image Description Evaluation) measures how well a generated caption matches a set of reference captions written by human annotators. It weights terms that are specific and informative to the image more heavily than common filler words, making it more meaningful than simple word overlap metrics.

### Why This Metric
CIDEr correlates more strongly with human judgment on captioning quality than BLEU. It captures whether the model describes what is actually distinctive about an image, which is exactly what degrades under heavy compression.

### Supporting Metrics
| Metric | Description |
|---|---|
| BLEU-4 | 4-gram precision overlap with reference captions |
| METEOR | Handles synonyms and paraphrasing, good secondary signal |
| ROUGE-L | Longest common subsequence overlap at sentence level |

---

## Visual Reasoning

**Dataset:** GQA  
**Primary Metric:** Accuracy  

### Description
GQA (Graph Question Answering) tests compositional and relational visual reasoning. Questions require multi-step logic such as identifying relationships between objects, understanding attributes, and chaining visual concepts. GQA is specifically designed to control for question bias present in earlier VQA datasets.

### Why This Metric
Visual reasoning tends to degrade earlier and more sharply than basic VQA under compression, making it a sensitive indicator of cognitive capability loss. GQA's structured question design also enables meaningful sub-metric analysis.

### Supporting Metrics
| Metric | Description |
|---|---|
| Consistency | Whether the model answers logically paired question sets consistently |
| Validity | Whether answers are a plausible type for the question asked |
| Plausibility | Whether answers make real-world sense |

---

## OCR / Document Understanding

**Dataset:** TextVQA  
**Primary Metric:** ANLS (Average Normalized Levenshtein Similarity)  

### Description
ANLS measures the similarity between a generated answer and the ground truth answer using normalized edit distance rather than exact match. A score of 1.0 means a perfect match. A score of 0.0 means the answer is completely wrong. Minor character-level differences such as spacing or punctuation are penalized proportionally rather than treated as complete failures.

### Why This Metric
TextVQA requires the model to read and reason about text that appears within images. Exact match accuracy is too harsh for OCR outputs where minor transcription differences are common. ANLS gives a fairer signal of reading comprehension capability under compression.

### Supporting Metrics
| Metric | Description |
|---|---|
| Exact Match Accuracy | Proportion of answers that exactly match ground truth |
| Partial Match Rate | Proportion of answers scoring above 0.5 ANLS |

---

## Hallucination

**Dataset:** COCO (using ground truth object annotations)  
**Primary Metric:** CHAIR-S (Sentence-level Caption Hallucination Assessment with Image Relevance)  

### Description
CHAIR measures how often a model mentions objects that are not actually present in the image. CHAIR-S computes the proportion of generated sentences that contain at least one hallucinated object mention. CHAIR-I computes the proportion of individual object mentions that are hallucinated.

### ⚠️ Important: Lower is Better
Unlike all other metric groups, a lower CHAIR-S score indicates better performance. An increase in CHAIR-S after compression means the model is hallucinating more frequently. The dashboard reflects this with inverted color coding, and the radar chart uses an inverted normalization for this group.

### Why This Metric
Hallucination is a critical trustworthiness signal for real-world VLM deployment. A compressed model that scores well on VQA but hallucinates heavily is not safe to deploy. LVLM-Compress-Bench specifically identified hallucination as one of the most important dimensions to track under compression.

### Supporting Metrics
| Metric | Description |
|---|---|
| CHAIR-I | Object-level hallucination rate |
| Coverage | Proportion of ground truth objects mentioned (prevents gaming by saying nothing) |

---

## Spatial Awareness

**Dataset:** VSR (Visual Spatial Reasoning)  
**Primary Metric:** Accuracy  

### Description
VSR is a binary true/false benchmark. Each sample consists of an image paired with a caption describing a spatial relationship between objects (e.g. "the cat is to the left of the dog"). The model must judge whether the stated relationship is true or false. The random baseline for VSR is 50%.

### Why This Metric
Spatial reasoning is a capability that requires fine-grained visual understanding and tends to be sensitive to compression of the vision encoder. Because of the 50% random baseline, raw accuracy alone is less meaningful — the gap above random is the more informative signal.

### Supporting Metrics
| Metric | Description |
|---|---|
| Above/Below Accuracy | Accuracy on vertical spatial relation questions |
| Left/Right Accuracy | Accuracy on horizontal spatial relation questions |
| Near/Far Accuracy | Accuracy on proximity spatial relation questions |
| Inside/Outside Accuracy | Accuracy on containment spatial relation questions |
| Gap Above Random | Raw accuracy minus 50% random baseline |

---

## Radar Chart Normalization

For the multi-group radar chart in the dashboard, all primary metrics are normalized to a **retention percentage** relative to the baseline model:
$$\text{Retention} = (compressed score / baseline score) \times 100$$

For CHAIR-S (lower is better), the normalization is inverted:
$$\text{Retention (\%)} = (compressed score / baseline score) \times 100$$

This ensures that for every group, a higher retention percentage consistently means the compressed model preserved more capability relative to baseline.