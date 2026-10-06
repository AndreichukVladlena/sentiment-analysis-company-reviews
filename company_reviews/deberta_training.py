"""Sequential five-fold validation and final DeBERTa training.

Run from the repository root: python -m company_reviews.deberta_training --mode all
"""

import argparse
import gc
import json
import os
import random
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from company_reviews.baseline import normalize_reviews_for_split
from company_reviews.training import read_training_data

CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "spm.model",
    "special_tokens_map.json",
    "training_config.json",
)


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_digest(config):
    payload = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return sha256(payload.encode()).hexdigest()[:12]


def training_config(data_path, clean, train_idx, valid_idx, device):
    """Keep the original fold configuration compatible with its verified checkpoint."""
    clean_hash = sha256()
    for review, rating in zip(clean["Review"], clean["Rating"], strict=True):
        clean_hash.update(review.encode("utf-8"))
        clean_hash.update(b"\0")
        clean_hash.update(bytes((int(rating),)))
    split_hash = sha256(
        train_idx.astype("<i8").tobytes() + valid_idx.astype("<i8").tobytes()
    ).hexdigest()
    config = {
        "data_sha256": file_hash(data_path),
        "clean_reviews_sha256": clean_hash.hexdigest(),
        "split_sha256": split_hash,
        "model_id": "microsoft/deberta-v3-base",
        "model_revision": "8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
        "max_length": 128,
        "use_fast_tokenizer": False,
        "batch_size": 8,
        "num_epochs": 1,
        "trainable_encoder_layers": 2,
        "encoder_lr": 3e-5,
        "head_lr": 1e-4,
        "warmup_fraction": 0.05,
        "max_grad_norm": 1.0,
        "optimizer": "AdamW",
        "loss": "cross_entropy",
        "seed": 2026,
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": version("transformers"),
    }
    if len(valid_idx) == 0:
        config.update(training_scope="full", train_rows=len(train_idx))
    return config


def checkpoint_path(cache_dir, config):
    scope = "full" if config.get("training_scope") == "full" else "top2"
    return Path(cache_dir) / f"deberta_v3_base_{scope}_{cache_digest(config)}"


def check_checkpoint(path, config):
    if not path.exists():
        return False
    if not all((path / name).is_file() for name in CHECKPOINT_FILES):
        raise RuntimeError(f"Checkpoint is incomplete: {path}")
    stored = json.loads((path / "training_config.json").read_text())
    if stored != config:
        raise RuntimeError(f"Checkpoint parameters do not match: {path}")
    return True


def probability_median(probabilities, rows):
    if (
        probabilities.shape != (rows, 5)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Invalid validation probabilities")
    return (probabilities.cumsum(axis=1) >= 0.5).argmax(axis=1) + 1


def fresh_model(config):
    """Reset initialization for every fold and for the final full-data fit."""
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        num_labels=5,
    )
    for parameter in model.deberta.parameters():
        parameter.requires_grad = False
    encoder = [
        parameter
        for layer in model.deberta.encoder.layer[-config["trainable_encoder_layers"] :]
        for parameter in layer.parameters()
    ]
    for parameter in encoder:
        parameter.requires_grad = True
    head = list(model.pooler.parameters()) + list(model.classifier.parameters())
    return model, encoder, head


def token_dataset(tokenizer, reviews, labels, config):
    tokens = tokenizer(
        reviews.tolist(),
        padding="max_length",
        truncation=True,
        max_length=config["max_length"],
        return_tensors="pt",
    )
    return TensorDataset(
        tokens["input_ids"],
        tokens["attention_mask"],
        torch.tensor(labels.to_numpy() - 1, dtype=torch.long),
    )


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def train_checkpoint(clean, train_idx, config, path):
    """Publish a completed checkpoint atomically; an interrupted epoch restarts."""
    if check_checkpoint(path, config):
        print(f"Сохранённая модель: {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        # Preserve an interrupted save for inspection instead of overwriting it.
        abandoned = temporary.with_name(temporary.name + f".{os.getpid()}")
        temporary.rename(abandoned)
        print(f"Незавершённое сохранение оставлено в {abandoned}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        use_fast=False,
    )
    dataset = token_dataset(
        tokenizer,
        clean["Review"].iloc[train_idx],
        clean["Rating"].iloc[train_idx],
        config,
    )
    model, encoder, head = fresh_model(config)
    model.to(config["device"]).train()
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder, "lr": config["encoder_lr"]},
            {"params": head, "lr": config["head_lr"]},
        ],
        foreach=False,
    )
    total_steps = len(loader) * config["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * config["warmup_fraction"]),
        num_training_steps=total_steps,
    )
    started = perf_counter()
    for epoch in range(config["num_epochs"]):
        loss_sum = 0.0
        for step, (input_ids, attention_mask, labels) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            output = model(
                input_ids=input_ids.to(config["device"]),
                attention_mask=attention_mask.to(config["device"]),
                labels=labels.to(config["device"]),
            )
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder + head, config["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            loss_sum += output.loss.item()
            if step % 500 == 0 or step == len(loader):
                print(
                    f"Эпоха {epoch + 1}, шаг {step}/{len(loader)}, "
                    f"loss {loss_sum / step:.4f}, "
                    f"{(perf_counter() - started) / 60:.1f} мин",
                    flush=True,
                )
    model.eval()
    temporary.mkdir()
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    atomic_json(temporary / "training_config.json", config)
    temporary.rename(path)
    print(
        f"Обучение завершено за {(perf_counter() - started) / 60:.1f} мин: {path}",
        flush=True,
    )


def validate_checkpoint(clean, valid_idx, config, path):
    if not check_checkpoint(path, config):
        raise RuntimeError(f"Checkpoint does not exist: {path}")
    weights_hash = file_hash(path / "model.safetensors")
    cache = path / f"valid_probabilities_{weights_hash[:12]}.npy"
    if cache.exists():
        probabilities = np.load(cache, allow_pickle=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, use_fast=False
        )
        dataset = token_dataset(
            tokenizer,
            clean["Review"].iloc[valid_idx],
            clean["Rating"].iloc[valid_idx],
            config,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True
        )
        model.to(config["device"]).eval()
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        batches = []
        with torch.inference_mode():
            for step, (input_ids, attention_mask, _) in enumerate(loader, 1):
                logits = model(
                    input_ids=input_ids.to(config["device"]),
                    attention_mask=attention_mask.to(config["device"]),
                ).logits
                batches.append(torch.softmax(logits, dim=1).cpu().numpy())
                if step % 100 == 0 or step == len(loader):
                    print(
                        f"Проверено {min(step * 32, len(dataset))}/{len(dataset)}",
                        flush=True,
                    )
        probabilities = np.concatenate(batches)
        probability_median(probabilities, len(valid_idx))
        temporary = cache.with_suffix(".npy.partial")
        with temporary.open("wb") as stream:
            np.save(stream, probabilities, allow_pickle=False)
        os.replace(temporary, cache)
    prediction = probability_median(probabilities, len(valid_idx))
    labels = clean["Rating"].iloc[valid_idx].to_numpy()
    errors = np.abs(labels - prediction)
    return {
        "rows": len(valid_idx),
        "mae": float(errors.mean()),
        "by_rating": {
            str(rating): {
                "rows": int((labels == rating).sum()),
                "mae": float(errors[labels == rating].mean()),
            }
            for rating in range(1, 6)
        },
        "checkpoint": str(path),
        "weights_sha256": weights_hash,
    }


def release_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run(
    data_path, cache_dir, mode="all", device="mps", dry_run=False, selected_folds=None
):
    if selected_folds is None:
        selected_folds = [1, 2, 3, 4, 5]
    if (
        not selected_folds
        or len(set(selected_folds)) != len(selected_folds)
        or any(fold not in range(1, 6) for fold in selected_folds)
    ):
        raise ValueError("Selected folds must be unique numbers from 1 through 5")
    clean = read_training_data(data_path)
    groups = normalize_reviews_for_split(clean["Review"])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    folds = list(splitter.split(clean["Review"], clean["Rating"], groups))
    configs = [training_config(data_path, clean, a, b, device) for a, b in folds]
    run_digest = cache_digest({"folds": configs})
    progress_path = Path(cache_dir) / f"deberta_cv_{run_digest}.json"
    progress = {"folds": [], "complete": False}
    if mode in {"cv", "all"}:
        for number, ((train_idx, valid_idx), config) in enumerate(
            zip(folds, configs, strict=True), 1
        ):
            if not set(groups.iloc[train_idx]).isdisjoint(groups.iloc[valid_idx]):
                raise ValueError(
                    "Review groups overlap between training and validation"
                )
            path = checkpoint_path(cache_dir, config)
            print(
                f"Фолд {number}: train={len(train_idx)}, valid={len(valid_idx)}, {path}",
                flush=True,
            )
            if dry_run:
                action = (
                    "обучение/проверка"
                    if number in selected_folds
                    else "только готовый кэш"
                )
                print(f"  {action}", flush=True)
                continue
            if number in selected_folds:
                train_checkpoint(clean, train_idx, config, path)
                release_memory()
            else:
                if not check_checkpoint(path, config):
                    continue
                digest = file_hash(path / "model.safetensors")[:12]
                if not (path / f"valid_probabilities_{digest}.npy").exists():
                    continue
            result = validate_checkpoint(clean, valid_idx, config, path)
            result["fold"] = number
            progress["folds"].append(result)
            print(f"MAE фолда {number}: {result['mae']:.8f}", flush=True)
            atomic_json(progress_path, progress)
            release_memory()
        if not dry_run:
            if len(progress["folds"]) == 5:
                scores = [item["mae"] for item in progress["folds"]]
                progress.update(
                    complete=True,
                    mean_mae=float(np.mean(scores)),
                    std_mae=float(np.std(scores)),
                    oof_mae=float(
                        np.average(scores, weights=[len(b) for _, b in folds])
                    ),
                )
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            atomic_json(progress_path, progress)
            print(json.dumps(progress, ensure_ascii=False, indent=2), flush=True)
        print(f"Результаты CV: {progress_path}", flush=True)
    if mode in {"full", "all"}:
        train_idx = np.arange(len(clean))
        config = training_config(
            data_path, clean, train_idx, np.array([], dtype=np.int64), device
        )
        path = checkpoint_path(cache_dir, config)
        print(f"Финальное обучение: {len(clean)} строк, {path}", flush=True)
        if not dry_run:
            train_checkpoint(clean, train_idx, config, path)
            release_memory()
    return progress_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--mode", choices=["cv", "full", "all"], default="all")
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        choices=range(1, 6),
        help="Train/validate selected CV folds; reuse other completed local folds",
    )
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; pass --device cpu explicitly")
    torch.set_num_threads(4)
    run(args.data, args.cache, args.mode, args.device, args.dry_run, args.folds)


if __name__ == "__main__":
    main()
