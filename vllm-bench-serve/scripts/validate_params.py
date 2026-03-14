#!/usr/bin/env python3
"""validate_params.py — Validate vllm bench serve parameter combinations.

Usage:
    python3 validate_params.py --backend openai-chat --endpoint /v1/chat/completions --dataset-name random [--random-input-len N] ...

Output: JSON {"valid": bool, "warnings": [...], "errors": [...]}
"""

import argparse
import json
import sys

# Backend → required --endpoint mapping (default is /v1/completions)
BACKEND_ENDPOINT_MAP = {
    "vllm": "/v1/completions",
    "openai": "/v1/completions",
    "openai-chat": "/v1/chat/completions",
    "openai-audio": "/v1/audio/transcriptions",
    "openai-embeddings": "/v1/embeddings",
    "openai-embeddings-chat": "/v1/embeddings",
    "openai-embeddings-clip": "/v1/embeddings",
    "openai-embeddings-vlm2vec": "/v1/embeddings",
    "infinity-embeddings": "/v1/embeddings",
    "infinity-embeddings-clip": "/v1/embeddings",
    "vllm-pooling": "/pooling",
    "vllm-rerank": "/v1/rerank",
}

# Dataset-backend compatibility matrix
DATASET_BACKEND_COMPAT = {
    "random": [
        "openai", "openai-chat", "vllm",
        "openai-embeddings", "openai-embeddings-chat",
        "openai-embeddings-clip", "openai-embeddings-vlm2vec",
        "infinity-embeddings", "infinity-embeddings-clip",
        "vllm-pooling",
    ],
    "random-mm": ["openai-chat"],
    "random-rerank": ["vllm-rerank"],
    "sharegpt": ["openai", "openai-chat", "vllm"],
    "burstgpt": ["openai", "openai-chat", "vllm"],
    "custom": ["openai", "openai-chat", "vllm"],
    "custom_mm": ["openai-chat"],
    "prefix_repetition": ["openai", "openai-chat", "vllm"],
    "spec_bench": ["openai", "openai-chat", "vllm"],
    "hf": None,  # varies by HF dataset, skip strict check
    "sonnet": ["openai", "openai-chat", "vllm"],
}

# Datasets that require --dataset-path
DATASETS_REQUIRING_PATH = {"sharegpt", "burstgpt", "custom", "custom_mm", "spec_bench"}

ALL_BACKENDS = [
    "vllm", "openai", "openai-chat", "openai-audio",
    "openai-embeddings", "openai-embeddings-chat",
    "openai-embeddings-clip", "openai-embeddings-vlm2vec",
    "infinity-embeddings", "infinity-embeddings-clip",
    "vllm-pooling", "vllm-rerank",
]

ALL_DATASETS = [
    "sharegpt", "burstgpt", "sonnet", "random", "random-mm",
    "random-rerank", "hf", "custom", "custom_mm",
    "prefix_repetition", "spec_bench",
]


def validate(args):
    errors = []
    warnings = []

    # Check backend is valid
    if args.backend not in ALL_BACKENDS:
        errors.append(f"Unknown backend: '{args.backend}'. Valid: {ALL_BACKENDS}")

    # Check dataset is valid
    if args.dataset_name not in ALL_DATASETS:
        errors.append(f"Unknown dataset: '{args.dataset_name}'. Valid: {ALL_DATASETS}")

    # Check dataset-backend compatibility
    compat = DATASET_BACKEND_COMPAT.get(args.dataset_name)
    if compat is not None and args.backend not in compat:
        errors.append(
            f"Dataset '{args.dataset_name}' is not compatible with backend '{args.backend}'. "
            f"Compatible backends: {compat}"
        )

    # Check dataset-path requirement
    if args.dataset_name in DATASETS_REQUIRING_PATH and not args.dataset_path:
        if args.dataset_name == "sharegpt":
            warnings.append(
                "ShareGPT dataset without --dataset-path: will attempt auto-download from HuggingFace."
            )
        else:
            errors.append(
                f"Dataset '{args.dataset_name}' requires --dataset-path."
            )

    # Check goodput format
    if args.goodput:
        valid_metrics = {"ttft", "tpot", "e2el"}
        for g in args.goodput:
            parts = g.split(":")
            if len(parts) != 2:
                errors.append(f"Invalid --goodput format: '{g}'. Expected 'METRIC:VALUE_MS'.")
            else:
                metric, value = parts
                if metric not in valid_metrics:
                    errors.append(
                        f"Invalid goodput metric: '{metric}'. Valid: {valid_metrics}"
                    )
                try:
                    float(value)
                except ValueError:
                    errors.append(f"Invalid goodput value: '{value}'. Must be a number (ms).")

    # Warn on low num_prompts
    if args.num_prompts and args.num_prompts < 100:
        warnings.append(
            f"--num-prompts={args.num_prompts} is low for P99 statistics. "
            f"Consider using at least 100 for meaningful percentile metrics."
        )

    # Check ramp-up consistency
    if args.ramp_up_strategy and args.request_rate:
        errors.append(
            "--ramp-up-strategy and --request-rate are mutually exclusive."
        )

    # Check --endpoint matches backend
    expected_endpoint = BACKEND_ENDPOINT_MAP.get(args.backend)
    if expected_endpoint and hasattr(args, "endpoint") and args.endpoint:
        if args.endpoint != expected_endpoint:
            errors.append(
                f"--endpoint '{args.endpoint}' does not match backend '{args.backend}'. "
                f"Expected: '{expected_endpoint}'"
            )
    elif expected_endpoint and expected_endpoint != "/v1/completions":
        # Backend needs non-default endpoint but none specified
        if not (hasattr(args, "endpoint") and args.endpoint):
            warnings.append(
                f"Backend '{args.backend}' requires --endpoint {expected_endpoint}. "
                f"The default /v1/completions will cause a URL validation error."
            )

    # Sonnet deprecation
    if args.dataset_name == "sonnet":
        warnings.append("Sonnet dataset is deprecated. Consider using 'random' instead.")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate vllm bench serve parameters")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--endpoint", default=None, help="API endpoint")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--input-len", type=int, default=None)
    parser.add_argument("--output-len", type=int, default=None)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--request-rate", type=float, default=None)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--goodput", nargs="*", default=None)
    parser.add_argument("--ramp-up-strategy", default=None)

    args = parser.parse_args()
    errors, warnings = validate(args)

    result = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }

    print(json.dumps(result, indent=2))
    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
