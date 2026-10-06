"""Локальная загрузка и HTTP-инференс на крошечной DeBERTa без скачивания весов."""

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import joblib
import pytest
import sklearn
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from company_reviews.api import create_app

torch = pytest.importorskip("torch")
sentencepiece = pytest.importorskip("sentencepiece")
transformers = pytest.importorskip("transformers")


@pytest.fixture
def tiny_checkpoint(tmp_path):
    directory = tmp_path / "models" / "deberta"
    directory.mkdir(parents=True)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Good service. Bad delivery. An ordinary review.\n" * 10)
    sentencepiece.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(tmp_path / "tiny"),
        vocab_size=32,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    tokenizer = transformers.DebertaV2Tokenizer(vocab_file=str(tmp_path / "tiny.model"))
    tokenizer.save_pretrained(directory)
    config = transformers.DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=5,
    )
    model = transformers.DebertaV2ForSequenceClassification(config)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([0.4, 0.05, 0.1, 0.05, 0.4]).log())
    model.save_pretrained(directory)
    spec = {
        "id": "deberta",
        "format": "deberta",
        "filename": "deberta",
        "version": "tiny-test",
        "file_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.iterdir()
        },
    }
    manifest = {"default_model": "deberta", "models": [spec]}
    (directory.parent / "manifest.json").write_text(json.dumps(manifest))
    return directory, tmp_path / "history.sqlite3"


def test_local_deberta_uses_cpu_truncation_minibatches_and_median(tiny_checkpoint):
    directory, db = tiny_checkpoint
    with TestClient(create_app(directory.parent, db)) as client:
        estimator = client.app.state.registry.snapshot().estimator
        batch_sizes = []
        lengths = []

        def check_forward(module, args, kwargs):
            inputs = kwargs["input_ids"]
            assert inputs.device.type == "cpu"
            assert not module.training
            assert not torch.is_grad_enabled()
            batch_sizes.append(len(inputs))
            lengths.append(inputs.shape[1])

        hook = estimator.model.register_forward_pre_hook(
            check_forward, with_kwargs=True
        )
        response = client.post("/predict", json=[{"Review": "Good " * 200}] * 17)
        hook.remove()
        assert response.status_code == 200
        assert batch_sizes == [8, 8, 1]
        assert lengths == [128, 128, 128]
        assert response.headers["X-Model-Id"] == "deberta"
        assert [row["label"] for row in response.json()] == [3] * 17
        assert [row["confidence"] for row in response.json()] == pytest.approx(
            [0.1] * 17
        )
        assert client.post("/predict", json={"Review": "Good"}).json()["label"] == 3
        assert (
            client.post("/load_model", json={"model_id": "deberta"}).status_code == 200
        )
        assert client.app.state.registry.snapshot().estimator is estimator


def test_switch_between_tfidf_and_deberta_records_artifact_identity(tiny_checkpoint):
    directory, db = tiny_checkpoint
    manifest_path = directory.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = directory.parent / "baseline.joblib"
    joblib.dump(DummyClassifier(strategy="prior").fit(["x"], [5]), path)
    manifest["models"].append(
        {
            "id": "baseline",
            "filename": path.name,
            "version": "test-baseline",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sklearn_version": sklearn.__version__,
        }
    )
    manifest["default_model"] = "baseline"
    manifest_path.write_text(json.dumps(manifest))
    with TestClient(create_app(directory.parent, db)) as client:
        assert client.get("/models").json()["active_model"] == "baseline"
        assert client.post("/predict", json={"Review": "Good"}).json()["label"] == 5
        assert (
            client.post("/load_model", json={"model_id": "deberta"}).status_code == 200
        )
        assert client.post("/predict", json={"Review": "Good"}).json()["label"] == 3
        assert (
            client.post("/load_model", json={"model_id": "baseline"}).status_code == 200
        )
        assert client.post("/predict", json={"Review": "Good"}).json()["label"] == 5
    fingerprint = hashlib.sha256(
        json.dumps(manifest["models"][0]["file_sha256"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    with sqlite3.connect(db) as connection:
        rows = connection.execute(
            "SELECT model_id, model_sha256 FROM requests ORDER BY rowid"
        ).fetchall()
    assert [row[0] for row in rows] == ["baseline", "deberta", "baseline"]
    assert rows[1][1] == fingerprint


def test_deberta_concurrent_batches_are_consistent(tiny_checkpoint):
    directory, db = tiny_checkpoint
    with (
        TestClient(create_app(directory.parent, db)) as client,
        ThreadPoolExecutor(max_workers=3) as executor,
    ):
        responses = list(
            executor.map(
                lambda _: client.post("/predict", json=[{"Review": "Good"}] * 9),
                range(3),
            )
        )
    assert all(response.status_code == 200 for response in responses)
    assert all(
        [row["label"] for row in response.json()] == [3] * 9 for response in responses
    )


@pytest.mark.parametrize("filename", ["model.safetensors", "config.json", "spm.model"])
def test_changed_deberta_file_preserves_previous_model(tiny_checkpoint, filename):
    directory, db = tiny_checkpoint
    with TestClient(create_app(directory.parent, db)) as client:
        (directory / filename).write_bytes(b"changed")
        response = client.post("/load_model", json={"model_id": "deberta"})
        assert response.status_code == 503
        assert (
            client.post("/predict", json={"Review": "Still usable"}).json()["label"]
            == 3
        )


@pytest.mark.parametrize(
    "problem", ["unhashed", "missing_hash", "outside_symlink", "four_labels"]
)
def test_deberta_rejects_incomplete_or_unsafe_checkpoint(tiny_checkpoint, problem):
    from company_reviews.model_registry import ModelLoadError, ModelRegistry

    directory, _ = tiny_checkpoint
    manifest_path = directory.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    hashes = manifest["models"][0]["file_sha256"]
    if problem == "unhashed":
        (directory / "tokenizer.json").write_text("{}")
    elif problem == "missing_hash":
        del hashes["config.json"]
    elif problem == "outside_symlink":
        target = directory.parent / "outside.model"
        (directory / "spm.model").rename(target)
        (directory / "spm.model").symlink_to(target)
    else:
        path = directory / "config.json"
        config = json.loads(path.read_text())
        config["id2label"].pop("4")
        config["label2id"].pop("LABEL_4")
        path.write_text(json.dumps(config))
        hashes["config.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ModelLoadError):
        ModelRegistry(directory.parent)


def test_exported_full_checkpoint_and_reexport_serve_predictions(
    tiny_checkpoint, tmp_path
):
    from company_reviews.export_deberta import export_deberta

    checkpoint, db = tiny_checkpoint
    (checkpoint / "training_config.json").write_text(
        json.dumps(
            {
                "training_scope": "full",
                "max_length": 128,
                "use_fast_tokenizer": False,
            }
        )
    )
    output = tmp_path / "exported"
    output.mkdir()
    (output / "manifest.json").write_bytes(
        (checkpoint.parent / "manifest.json").read_bytes()
    )
    for _ in range(2):
        export_deberta(checkpoint, output)
        with TestClient(create_app(output, db)) as client:
            result = client.post("/predict", json={"Review": "Good service"})
            assert result.status_code == 200
            assert result.json()["label"] == 3
            assert result.json()["confidence"] == pytest.approx(0.1)
