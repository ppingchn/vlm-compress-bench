"""
General efficiency metrics evaluator for Visual Language Models.

This module measures hardware and efficiency characteristics of VLMs directly
without requiring dataset evaluation. Key metrics measured include:

- Model size in megabytes (MB)
- Parameter count
- Compression ratio relative to baseline model size
- End-to-end inference latency in milliseconds (ms)
- Peak GPU memory allocation in megabytes (MB)
- Inference throughput in tokens per second
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import torch
from tqdm import tqdm

from benchmarker.utils import generate_text

logger = logging.getLogger(__name__)


def _is_cuda(device: str) -> bool:
    """Check whether the given device string specifies an available CUDA device.

    Args:
        device: Torch device string (e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``).

    Returns:
        True if PyTorch CUDA is available and the device is a CUDA device,
        False otherwise.
    """
    if not torch.cuda.is_available():
        return False
    try:
        dev = torch.device(device)
        return dev.type == "cuda"
    except Exception:
        return str(device).startswith("cuda")


def measure_model_size(model: Any) -> float:
    """Measure total model parameter memory size in megabytes (MB).

    Computes the sum of ``param.nelement() * param.element_size()`` across all
    parameters in the model and converts the total bytes to megabytes.

    Args:
        model: A PyTorch model or object with a ``parameters()`` method.

    Returns:
        Model size in megabytes (MB). Returns 0.0 if parameter inspection fails.
    """
    try:
        total_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
        return float(total_bytes / (1024.0 * 1024.0))
    except Exception as exc:
        logger.warning("Failed to compute model size: %s", exc)
        return 0.0


def count_parameters(model: Any) -> int:
    """Count the total number of parameters in the model.

    Args:
        model: A PyTorch model or object with a ``parameters()`` method.

    Returns:
        Total number of parameters as an integer. Returns 0 if counting fails.
    """
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception as exc:
        logger.warning("Failed to count model parameters: %s", exc)
        return 0


def measure_inference_latency(
    model: Any,
    processor: Any,
    *,
    image: Any,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 128,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> float:
    """Measure average end-to-end inference latency in milliseconds (ms).

    Performs *num_warmup* un-timed initial generation calls to warm up hardware
    and CUDA caches, followed by *num_runs* timed generation calls. When running
    on a CUDA device, synchronises the CUDA context before and after timing each
    run.

    Args:
        model: HuggingFace VLM with a ``generate()`` method.
        processor: Paired processor for image + prompt inputs.
        image: Image input (PIL Image or tensor).
        prompt: Text prompt string.
        device: Torch device string (e.g. ``"cuda"``, ``"cpu"``).
        max_new_tokens: Maximum tokens to generate per call.
        num_warmup: Number of warmup inference iterations.
        num_runs: Number of timed inference iterations.

    Returns:
        Average inference latency in milliseconds (ms). Returns 0.0 if num_runs <= 0
        or if all runs fail.
    """
    if num_runs <= 0:
        return 0.0

    is_cuda_device = _is_cuda(device)

    with torch.no_grad():
        # Warmup iterations
        for _ in range(max(0, num_warmup)):
            try:
                generate_text(
                    model,
                    processor,
                    image=image,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as exc:
                logger.warning("Latency warmup iteration failed: %s", exc)

        # Timed runs
        latencies_ms: list[float] = []
        for _ in tqdm(range(num_runs), desc="Measuring Latency", leave=False):
            try:
                if is_cuda_device:
                    torch.cuda.synchronize(device)
                t_start = time.perf_counter()

                generate_text(
                    model,
                    processor,
                    image=image,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=max_new_tokens,
                )

                if is_cuda_device:
                    torch.cuda.synchronize(device)
                t_end = time.perf_counter()

                latencies_ms.append((t_end - t_start) * 1000.0)
            except Exception as exc:
                logger.warning("Inference latency run failed: %s", exc)

    if not latencies_ms:
        return 0.0

    avg_latency = sum(latencies_ms) / len(latencies_ms)
    return float(avg_latency)


def measure_gpu_memory(
    model: Any,
    processor: Any,
    *,
    image: Any,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 128,
) -> float:
    """Measure peak GPU memory allocated during inference in megabytes (MB).

    Resets the peak GPU memory statistics, runs a single generation pass,
    and queries the maximum memory allocated on the CUDA device. Returns 0.0
    if not running on CUDA or if memory measurement fails.

    Args:
        model: HuggingFace VLM with a ``generate()`` method.
        processor: Paired processor for image + prompt inputs.
        image: Image input (PIL Image or tensor).
        prompt: Text prompt string.
        device: Torch device string (e.g. ``"cuda"``, ``"cpu"``).
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Peak memory allocated in megabytes (MB) as a float. Returns 0.0 on CPU.
    """
    if not _is_cuda(device):
        return 0.0

    try:
        cuda_device = torch.device(device)
        torch.cuda.reset_peak_memory_stats(cuda_device)

        with torch.no_grad():
            generate_text(
                model,
                processor,
                image=image,
                prompt=prompt,
                device=device,
                max_new_tokens=max_new_tokens,
            )

        peak_bytes = torch.cuda.max_memory_allocated(cuda_device)
        return float(peak_bytes / (1024.0 * 1024.0))
    except Exception as exc:
        logger.warning("Failed to measure GPU memory: %s", exc)
        return 0.0


def measure_throughput(
    model: Any,
    processor: Any,
    *,
    image: Any,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 128,
    num_runs: int = 5,
) -> float:
    """Measure inference throughput in generated tokens per second.

    Runs *num_runs* generation passes, counting the total number of newly generated
    tokens and dividing by the total execution time across all runs.

    Args:
        model: HuggingFace VLM with a ``generate()`` method.
        processor: Paired processor for image + prompt inputs.
        image: Image input (PIL Image or tensor).
        prompt: Text prompt string.
        device: Torch device string (e.g. ``"cuda"``, ``"cpu"``).
        max_new_tokens: Maximum tokens to generate per call.
        num_runs: Number of timed generation iterations.

    Returns:
        Throughput in tokens per second as a float. Returns 0.0 if timing or token count is 0.
    """
    if num_runs <= 0:
        return 0.0

    is_cuda_device = _is_cuda(device)

    try:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
    except Exception as exc:
        logger.warning("Failed to prepare inputs for throughput measurement: %s", exc)
        return 0.0

    total_tokens = 0
    total_time_sec = 0.0

    with torch.no_grad():
        for _ in tqdm(range(num_runs), desc="Measuring Throughput", leave=False):
            try:
                if is_cuda_device:
                    torch.cuda.synchronize(device)
                t_start = time.perf_counter()

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

                if is_cuda_device:
                    torch.cuda.synchronize(device)
                t_end = time.perf_counter()

                gen_tokens = max(0, output_ids.shape[-1] - input_len)
                total_tokens += gen_tokens
                total_time_sec += (t_end - t_start)
            except Exception as exc:
                logger.warning("Throughput iteration failed: %s", exc)

    if total_time_sec <= 0.0 or total_tokens == 0:
        return 0.0

    return float(total_tokens / total_time_sec)


def compute_compression_ratio(
    baseline_size_mb: Optional[float],
    compressed_size_mb: float,
) -> float:
    """Compute the compression ratio between baseline and compressed model size.

    Compression ratio = baseline_size_mb / compressed_size_mb.

    Args:
        baseline_size_mb: Size of the uncompressed baseline model in MB.
        compressed_size_mb: Size of the compressed model in MB.

    Returns:
        Compression ratio as a float. Returns 1.0 if compressed_size_mb is 0 (or <= 0)
        or if baseline_size_mb is None or <= 0.
    """
    if compressed_size_mb <= 0.0 or baseline_size_mb is None or baseline_size_mb <= 0.0:
        return 1.0
    return float(baseline_size_mb / compressed_size_mb)


def evaluate_general_efficiency(
    model: Any,
    processor: Any,
    *,
    image: Any,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 128,
    baseline_size_mb: Optional[float] = None,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> Dict[str, Any]:
    """Run all general efficiency measurements and return a schema-conforming dict.

    Measures model size, parameter count, inference latency, peak GPU memory,
    throughput, and compression ratio.

    Args:
        model: HuggingFace-compatible VLM.
        processor: Paired processor for model inputs.
        image: Test image for benchmark runs.
        prompt: Test prompt string for benchmark runs.
        device: Torch device string (e.g. ``"cuda"``, ``"cpu"``).
        max_new_tokens: Maximum tokens to generate per inference call.
        baseline_size_mb: Model size of baseline in MB (used for compression ratio).
        num_warmup: Warmup iterations for latency measurement.
        num_runs: Timed iterations for latency measurement.

    Returns:
        A dictionary containing:
            - ``model_size_mb`` (float)
            - ``parameter_count`` (int)
            - ``compression_ratio`` (float)
            - ``inference_latency_ms`` (float)
            - ``gpu_memory_mb`` (float)
            - ``throughput_tokens_per_sec`` (float)
    """
    logger.info("Evaluating general efficiency metrics on device '%s'…", device)

    model_size_mb = measure_model_size(model)
    parameter_count = count_parameters(model)

    comp_ratio_baseline = baseline_size_mb if baseline_size_mb is not None else model_size_mb
    compression_ratio = compute_compression_ratio(comp_ratio_baseline, model_size_mb)

    inference_latency_ms = measure_inference_latency(
        model,
        processor,
        image=image,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        num_warmup=num_warmup,
        num_runs=num_runs,
    )

    gpu_memory_mb = measure_gpu_memory(
        model,
        processor,
        image=image,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
    )

    throughput_tokens_per_sec = measure_throughput(
        model,
        processor,
        image=image,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        num_runs=min(num_runs, 5),
    )

    results = {
        "model_size_mb": float(model_size_mb),
        "parameter_count": int(parameter_count),
        "compression_ratio": float(compression_ratio),
        "inference_latency_ms": float(inference_latency_ms),
        "gpu_memory_mb": float(gpu_memory_mb),
        "throughput_tokens_per_sec": float(throughput_tokens_per_sec),
    }

    logger.info("General efficiency evaluation complete — %s", results)
    return results
