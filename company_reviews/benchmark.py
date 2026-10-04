"""Measure local CPU inference, optionally including cached transformer experiments.

python -m company_reviews.benchmark --advanced
Advanced models require the optional advanced dependencies and existing local caches.
This replays historical notebook evaluation caches and measures fresh CPU latency;
it does not perform a new independent validation pass.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from company_reviews.baseline import (
    median_from_probabilities,
    normalize_reviews_for_split,
)
from company_reviews.training import file_sha256, read_training_data


def resolve_cache_path(explicit, root, pattern, option):
    """Accept an explicit artifact or require exactly one matching cached run."""
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(
                f"{option}: {path} does not exist; run the corresponding notebook first"
            )
        return path

    matches = sorted(Path(root).glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No {pattern} cache; run the corresponding notebook first or supply {option}"
        )
    if len(matches) != 1:
        raise ValueError(
            f"Multiple cached experiments match {pattern}; select one with {option}: {matches}"
        )
    return matches[0]


def measure(predict, samples, repeats):
    timings = []
    by_length = []
    for text in samples:
        predict([text])  # one warmup per length
        runs = []
        for _ in range(repeats):
            started = perf_counter()
            predict([text])
            runs.append((perf_counter() - started) * 1000)
        timings.extend(runs)
        by_length.append(
            {
                "characters": len(text),
                "median_ms": float(np.median(runs)),
                "p95_ms": float(np.percentile(runs, 95)),
            }
        )
    return {
        "requests": len(timings),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "by_length": by_length,
    }


def benchmark(
    *,
    advanced=False,
    repeats=10,
    threads=2,
    output=Path("models/benchmark.json"),
    deberta_dir=None,
    mpnet_manifest=None,
    catboost_manifest=None,
):
    data_path = Path("data/raw/train.csv")
    data = read_training_data(data_path)
    lengths = data["Review"].str.len()
    samples = [
        data.loc[(lengths - lengths.quantile(q)).abs().idxmin(), "Review"]
        for q in (0.1, 0.5, 0.9)
    ]
    groups = normalize_reviews_for_split(data["Review"])
    train_idx, valid_idx = next(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026).split(
            data["Review"], data["Rating"], groups
        )
    )
    split_hash = sha256(
        train_idx.astype("<i8").tobytes() + valid_idx.astype("<i8").tobytes()
    ).hexdigest()
    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "threads": threads,
        "batch_size": 1,
        "repeats_per_length": repeats,
        "sample_length_quantiles": [0.1, 0.5, 0.9],
        "scope": "Warm direct model calls including text preprocessing; excludes HTTP/network and cold startup.",
        "data_sha256": file_sha256(data_path),
        "split_sha256": split_hash,
        "models": {},
    }
    manifest = json.loads(Path("models/manifest.json").read_text())
    artifacts = {entry["id"]: entry for entry in manifest["models"]}

    with threadpool_limits(limits=threads):
        for model_id in ("tfidf", "word-only"):
            entry = artifacts[model_id]
            path = Path("models") / entry["filename"]
            artifact_hash = file_sha256(path)
            if artifact_hash != entry["sha256"]:
                raise ValueError(f"Artifact hash does not match manifest: {model_id}")
            started = perf_counter()
            model = joblib.load(path)
            load_seconds = perf_counter() - started
            report["models"][model_id] = {
                "load_seconds": load_seconds,
                "artifact_bytes": path.stat().st_size,
                "artifact_sha256": artifact_hash,
                "version": entry["version"],
                **measure(model.predict_proba, samples, repeats),
            }
            del model
            gc.collect()
        if advanced:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            torch.set_num_threads(threads)
            model_dir = resolve_cache_path(
                deberta_dir,
                Path("data/cache"),
                "deberta*/training_config.json",
                "--deberta-dir",
            )
            if deberta_dir is None:
                model_dir = model_dir.parent
            config = json.loads((model_dir / "training_config.json").read_text())
            if (
                config["data_sha256"] != report["data_sha256"]
                or config["split_sha256"] != split_hash
            ):
                raise ValueError(
                    "DeBERTa cached experiment provenance differs from current data or split"
                )
            started = perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir, use_fast=False, local_files_only=True
            )
            model = (
                AutoModelForSequenceClassification.from_pretrained(
                    model_dir, local_files_only=True
                )
                .cpu()
                .eval()
            )
            load_seconds = perf_counter() - started

            def predict_deberta(reviews):
                tokens = tokenizer(
                    reviews,
                    padding=True,
                    truncation=True,
                    max_length=config["max_length"],
                    return_tensors="pt",
                )
                with torch.inference_mode():
                    return torch.softmax(model(**tokens).logits, dim=1).numpy()

            weights_hash = file_sha256(model_dir / "model.safetensors")
            probabilities = np.load(
                model_dir / f"valid_probabilities_{weights_hash[:12]}.npy"
            )
            mae = mean_absolute_error(
                data["Rating"].iloc[valid_idx],
                median_from_probabilities(probabilities, np.arange(1, 6)),
            )
            report["models"]["deberta-experiment"] = {
                "load_seconds": load_seconds,
                "artifact_bytes": sum(
                    p.stat().st_size for p in model_dir.iterdir() if p.is_file()
                ),
                "max_tokens": config["max_length"],
                "model_directory": str(model_dir),
                "training_device": config["device"],
                "evaluation_source": "Historical notebook validation probabilities; no fresh CPU validation pass",
                "first_fold_mae": mae,
                "weights_sha256": weights_hash,
                **measure(predict_deberta, samples, repeats),
            }
            del model, tokenizer, probabilities
            gc.collect()
            from catboost import CatBoostRegressor
            from sentence_transformers import SentenceTransformer

            mpnet_path = resolve_cache_path(
                mpnet_manifest, Path("data/cache"), "mpnet*.json", "--mpnet-manifest"
            )
            cat_path = resolve_cache_path(
                catboost_manifest,
                Path("data/cache"),
                "catboost_mpnet*.json",
                "--catboost-manifest",
            )
            mpnet_config = json.loads(mpnet_path.read_text())
            cat_config = json.loads(cat_path.read_text())
            if mpnet_path.stem != f"mpnet_{cat_config['embedding_digest']}":
                raise ValueError(
                    "CatBoost manifest references another MPNet embedding cache"
                )
            if (
                cat_config["data_sha256"] != report["data_sha256"]
                or cat_config["split_sha256"] != split_hash
            ):
                raise ValueError(
                    "MPNet cached experiment provenance differs from current data or split"
                )
            started = perf_counter()
            encoder = SentenceTransformer(
                mpnet_config["model_id"],
                revision=mpnet_config["model_revision"],
                device="cpu",
                local_files_only=True,
            )
            encoder.max_seq_length = mpnet_config["max_length"]
            regressor = CatBoostRegressor(thread_count=threads)
            regressor.load_model(str(cat_path.with_suffix(".cbm")))
            load_seconds = perf_counter() - started

            def predict_mpnet(reviews):
                vectors = encoder.encode(
                    reviews,
                    normalize_embeddings=mpnet_config["normalize_embeddings"],
                    show_progress_bar=False,
                )
                return np.clip(regressor.predict(vectors, thread_count=threads), 1, 5)

            embeddings = np.load(mpnet_path.with_suffix(".npy"), mmap_mode="r")
            cached_prediction = np.clip(
                regressor.predict(embeddings[valid_idx], thread_count=threads), 1, 5
            )
            report["models"]["mpnet-catboost-experiment"] = {
                "load_seconds": load_seconds,
                "max_tokens": mpnet_config["max_length"],
                "mpnet_manifest": str(mpnet_path),
                "catboost_manifest": str(cat_path),
                "embedding_sha256": file_sha256(mpnet_path.with_suffix(".npy")),
                "regressor_sha256": file_sha256(cat_path.with_suffix(".cbm")),
                "embedding_device": mpnet_config["device"],
                "evaluation_source": "Historical notebook embeddings scored by cached regressor; no fresh CPU encoder validation pass",
                "first_fold_mae": mean_absolute_error(
                    data["Rating"].iloc[valid_idx], cached_prediction
                ),
                **measure(predict_mpnet, samples, repeats),
            }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--advanced", action="store_true")
    parser.add_argument(
        "--deberta-dir",
        type=Path,
        help="Directory containing DeBERTa training_config.json and weights",
    )
    parser.add_argument(
        "--mpnet-manifest",
        type=Path,
        help="MPNet .json manifest; embeddings use the same stem plus .npy",
    )
    parser.add_argument(
        "--catboost-manifest",
        type=Path,
        help="CatBoost .json manifest; model uses the same stem plus .cbm",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("models/benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error("repeats and threads must be positive")
    benchmark(
        advanced=args.advanced,
        repeats=args.repeats,
        threads=args.threads,
        output=args.output,
        deberta_dir=args.deberta_dir,
        mpnet_manifest=args.mpnet_manifest,
        catboost_manifest=args.catboost_manifest,
    )


if __name__ == "__main__":
    main()
