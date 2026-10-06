"""Small checks for CV isolation, checkpoint reuse and ordinal scoring."""

import json
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from company_reviews import deberta_training as training


def test_clean_hash_and_split_identity_follow_actual_rows(tmp_path):
    data = pd.DataFrame({"Review": ["Good", "Bad"], "Rating": [5, 1]})
    path = tmp_path / "train.csv"
    data.to_csv(path, index=False)
    first = training.training_config(path, data, np.array([0]), np.array([1]), "cpu")
    changed = data.copy()
    changed.loc[0, "Review"] = "Great"
    second = training.training_config(
        path, changed, np.array([0]), np.array([1]), "cpu"
    )
    reverse = training.training_config(path, data, np.array([1]), np.array([0]), "cpu")
    assert first["clean_reviews_sha256"] != sha256().hexdigest()
    assert first["clean_reviews_sha256"] != second["clean_reviews_sha256"]
    assert first["split_sha256"] != reverse["split_sha256"]


def test_probability_median_rejects_corrupt_cache():
    probabilities = np.array([[0.4, 0.0, 0.2, 0.0, 0.4], [0, 0, 0, 0, 1]])
    np.testing.assert_array_equal(training.probability_median(probabilities, 2), [3, 5])
    for invalid in [probabilities[:1], probabilities * 2, probabilities * np.nan]:
        with pytest.raises(ValueError):
            training.probability_median(invalid, 2)


def test_checkpoint_requires_complete_matching_files(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    assert not training.check_checkpoint(checkpoint, {"seed": 2026})
    checkpoint.mkdir()
    with pytest.raises(RuntimeError, match="incomplete"):
        training.check_checkpoint(checkpoint, {"seed": 2026})
    for filename in training.CHECKPOINT_FILES:
        (checkpoint / filename).write_text("test")
    manifest = checkpoint / "training_config.json"
    manifest.write_text(json.dumps({"seed": 2026}))
    assert training.check_checkpoint(checkpoint, {"seed": 2026})
    with pytest.raises(RuntimeError, match="match"):
        training.check_checkpoint(checkpoint, {"seed": 0})


def test_fold_training_starts_pretrained_and_freezes_lower_layers(monkeypatch):
    from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

    config = DebertaV2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=3,
        num_labels=5,
    )
    sources = []

    def load(source, **kwargs):
        sources.append((source, kwargs["revision"]))
        return DebertaV2ForSequenceClassification(config)

    monkeypatch.setattr(
        training.AutoModelForSequenceClassification, "from_pretrained", load
    )
    settings = {
        "seed": 2026,
        "model_id": "original",
        "model_revision": "pinned",
        "trainable_encoder_layers": 2,
    }
    first, _, _ = training.fresh_model(settings)
    original = first.classifier.weight.detach().clone()
    first.classifier.weight.data.fill_(99)
    second, encoder, head = training.fresh_model(settings)
    assert sources == [("original", "pinned"), ("original", "pinned")]
    assert second.classifier.weight.equal(original)
    assert not any(
        p.requires_grad for p in second.deberta.encoder.layer[0].parameters()
    )
    assert all(p.requires_grad for p in encoder + head)
    assert len(encoder + head) == sum(p.requires_grad for p in second.parameters())


def test_selected_folds_reuse_available_scores_without_training_omitted(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "Id": range(100),
            "Review": [f"review {i}" for i in range(100)],
            "Rating": np.tile(np.arange(1, 6), 20),
        }
    ).to_csv(data_path, index=False)
    clean = training.read_training_data(data_path)
    groups = training.normalize_reviews_for_split(clean["Review"])
    folds = list(
        training.StratifiedGroupKFold(5, shuffle=True, random_state=2026).split(
            clean["Review"],
            clean["Rating"],
            groups,
        )
    )
    cache = tmp_path / "cache"
    paths = []

    def save_complete(config, path, valid_idx):
        path.mkdir(parents=True, exist_ok=True)
        for name in training.CHECKPOINT_FILES:
            (path / name).write_text("test")
        (path / "training_config.json").write_text(json.dumps(config))
        digest = training.file_hash(path / "model.safetensors")[:12]
        np.save(
            path / f"valid_probabilities_{digest}.npy",
            np.full((len(valid_idx), 5), 0.2),
        )

    for train_idx, valid_idx in folds:
        config = training.training_config(data_path, clean, train_idx, valid_idx, "cpu")
        paths.append(training.checkpoint_path(cache, config))
    first_config = training.training_config(data_path, clean, *folds[0], "cpu")
    save_complete(first_config, paths[0], folds[0][1])
    trained = []

    def train(clean, train_idx, config, path):
        number = paths.index(path) + 1
        trained.append(number)
        save_complete(config, path, folds[number - 1][1])

    monkeypatch.setattr(training, "train_checkpoint", train)
    monkeypatch.setattr(training, "release_memory", lambda: None)
    progress_path = training.run(
        data_path, cache, mode="cv", device="cpu", selected_folds=[3, 4, 5]
    )
    progress = json.loads(progress_path.read_text())
    assert trained == [3, 4, 5]
    assert [item["fold"] for item in progress["folds"]] == [1, 3, 4, 5]
    assert progress["complete"] is False
    assert "mean_mae" not in progress and "oof_mae" not in progress

    trained.clear()
    training.run(data_path, cache, mode="cv", device="cpu", selected_folds=[2])
    progress = json.loads(progress_path.read_text())
    assert trained == [2]
    assert progress["complete"] is True
    assert len(progress["folds"]) == 5
    assert progress["mean_mae"] == pytest.approx(
        np.mean([item["mae"] for item in progress["folds"]])
    )


@pytest.mark.parametrize("folds", [[], [0], [6], [1, 1]])
def test_selected_folds_reject_invalid_numbers_before_reading_data(tmp_path, folds):
    with pytest.raises(ValueError, match="fold"):
        training.run(tmp_path / "missing.csv", tmp_path, selected_folds=folds)
