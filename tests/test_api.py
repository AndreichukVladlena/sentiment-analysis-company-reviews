"""HTTP contract, safe model replacement and persistent request history."""

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import pytest
import sklearn
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from company_reviews import api


@pytest.fixture
def service_files(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    models = []
    for name, labels in [("first", [1, 2, 2, 5, 5]), ("second", [1, 5, 5, 5])]:
        model = DummyClassifier(strategy="prior").fit(["x"] * len(labels), labels)
        path = model_dir / f"{name}.joblib"
        joblib.dump(model, path)
        models.append(
            {
                "id": name,
                "filename": path.name,
                "version": f"test-{name}",
                "description": "Тестовая модель",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sklearn_version": sklearn.__version__,
            }
        )
    (model_dir / "manifest.json").write_text(
        json.dumps(
            {
                "default_model": "first",
                "models": models,
            }
        )
    )
    return model_dir, tmp_path / "history.sqlite3"


@pytest.fixture
def client(service_files):
    model_dir, db = service_files
    with TestClient(api.create_app(model_dir=model_dir, history_db=db)) as client:
        yield client


def test_single_and_batch_return_median_and_its_probability(client):
    row = {"Id": 12, "Review": "Отличная доставка 📦"}
    expected = {"label": 2, "confidence": 0.4}
    assert client.post("/predict", json=row).json() == expected
    assert client.post("/predict", json=[row, {"Review": "Awful!"}]).json() == [
        expected,
        expected,
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"Review": ""},
        {"Review": " \t\n "},
        {"Review": 42},
        {"Review": None},
        {"Review": "x" * 20_001},
        {"Review": "x", "Id": True},
        {"Review": "x", "Id": "1"},
        {"Review": "x", "Id": 2**63},
        {"Review": "x", "Rating": 5},
        [],
        [{"Review": "x"}, {"Review": " "}],
        [{"Review": "x"}] * 129,
        [{"Review": "x" * 20_000}] * 11,
        "review",
        None,
    ],
)
def test_invalid_input_rejected_without_echoing_reviews(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"]
    assert "input" not in response.text


def test_payload_limit_applies_before_json_parsing(client):
    response = client.post("/predict", content=b" " * 1_048_577)
    assert response.status_code == 413


def test_model_selection_and_failed_load_preserve_active_model(client, service_files):
    assert client.get("/health").json()["model_id"] == "first"
    assert len(client.get("/models").json()["models"]) == 2
    assert client.post("/load_model", json={"model_id": "second"}).status_code == 200
    assert client.post("/predict", json={"Review": "Hello"}).json() == {
        "label": 5,
        "confidence": 0.75,
    }
    model_dir, _ = service_files
    (model_dir / "first.joblib").write_bytes(b"tampered artifact")
    response = client.post("/load_model", json={"model_id": "first"})
    assert response.status_code == 503
    assert client.get("/health").json()["model_id"] == "second"
    assert client.post("/load_model", json={"model_id": "unknown"}).status_code == 404
    assert (
        client.post("/load_model", json={"model_id": "../../file"}).status_code == 422
    )
    assert (
        client.post("/load_model", json={"url": "https://example.org/x"}).status_code
        == 422
    )


def test_history_persists_model_version_and_omits_raw_text(service_files):
    model_dir, db = service_files
    with TestClient(api.create_app(model_dir=model_dir, history_db=db)) as client:
        client.post(
            "/predict", json=[{"Id": 10, "Review": "PRIVATE TEXT"}, {"Review": "Other"}]
        )
        client.post("/load_model", json={"model_id": "second"})
        client.post("/predict", json={"Review": "New"})
    with TestClient(api.create_app(model_dir=model_dir, history_db=db)) as client:
        assert client.get("/health").status_code == 200
    with sqlite3.connect(db) as connection:
        requests = connection.execute(
            "SELECT model_id, model_version, item_count FROM requests ORDER BY rowid"
        ).fetchall()
        rows = connection.execute(
            "SELECT dataset_id, review_text, review_sha256, review_length, label FROM predictions ORDER BY rowid"
        ).fetchall()
    assert requests == [("first", "test-first", 2), ("second", "test-second", 1)]
    assert rows[0] == (10, None, hashlib.sha256(b"PRIVATE TEXT").hexdigest(), 12, 2)
    assert len(rows) == 3


def test_concurrent_requests_and_model_swaps_use_one_model_per_batch(
    client, service_files
):
    def operation(index):
        if index % 3 == 0:
            return client.post(
                "/load_model", json={"model_id": "second" if index % 2 else "first"}
            )
        response = client.post("/predict", json=[{"Review": "Good"}] * 5)
        assert len({(row["label"], row["confidence"]) for row in response.json()}) == 1
        return response

    with ThreadPoolExecutor(max_workers=6) as executor:
        responses = list(executor.map(operation, range(24)))
    assert all(response.status_code == 200 for response in responses)
    with sqlite3.connect(service_files[1]) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 80
        )
        inconsistent = connection.execute("""
            SELECT COUNT(*) FROM predictions p JOIN requests r USING (request_id)
            WHERE (r.model_id = 'first' AND p.label != 2)
               OR (r.model_id = 'second' AND p.label != 5)
        """).fetchone()[0]
    assert inconsistent == 0


def test_swagger_exposes_russian_examples_and_contract(client):
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/predict"]["post"]
    assert "confidence" in operation["description"]
    assert operation["requestBody"]["content"]["application/json"]["examples"][
        "single"
    ]["value"]["Review"]


def test_model_change_during_inference_keeps_batch_and_history_on_old_model(
    client, service_files, monkeypatch
):
    from threading import Event

    started = Event()
    finish = Event()
    active = client.app.state.registry.snapshot()
    original = active.estimator.predict_proba

    def blocking_predict(reviews):
        started.set()
        assert finish.wait(timeout=10)
        return original(reviews)

    monkeypatch.setattr(active.estimator, "predict_proba", blocking_predict)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(
            client.post, "/predict", json=[{"Review": "Review"}] * 2
        )
        try:
            assert started.wait(timeout=10)
            assert (
                client.post("/load_model", json={"model_id": "second"}).status_code
                == 200
            )
        finally:
            finish.set()
        response = pending.result(timeout=10)
    assert response.json() == [{"label": 2, "confidence": 0.4}] * 2
    assert response.headers["X-Model-Id"] == "first"
    assert client.post("/predict", json={"Review": "Next"}).json()["label"] == 5
    with sqlite3.connect(service_files[1]) as connection:
        assert connection.execute(
            "SELECT model_id FROM requests ORDER BY rowid"
        ).fetchall() == [("first",), ("second",)]


def test_history_failure_rolls_back_entire_batch_and_service_recovers(
    client, service_files
):
    with sqlite3.connect(service_files[1]) as connection:
        connection.executescript("""
            CREATE TRIGGER fail_second BEFORE INSERT ON predictions
            WHEN NEW.item_index = 1
            BEGIN SELECT RAISE(ABORT, 'Simulated unavailable storage'); END;
        """)
    response = client.post("/predict", json=[{"Review": "A"}, {"Review": "B"}])
    assert response.status_code == 503
    with sqlite3.connect(service_files[1]) as connection:
        assert connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0
        connection.execute("DROP TRIGGER fail_second")
    assert client.post("/predict", json={"Review": "Retry"}).status_code == 200


def test_inference_failure_does_not_clear_active_model(client, monkeypatch):
    active = client.app.state.registry.snapshot()
    with monkeypatch.context() as patch:
        patch.setattr(
            active.estimator,
            "predict_proba",
            lambda reviews: np.full((len(reviews), 3), np.nan),
        )
        assert client.post("/predict", json={"Review": "Review"}).status_code == 503
        assert client.get("/health").json()["model_id"] == "first"
    assert client.post("/predict", json={"Review": "Retry"}).json() == {
        "label": 2,
        "confidence": 0.4,
    }


def test_loading_invalid_distribution_does_not_replace_active_model(service_files):
    model_dir, db = service_files
    model = DummyClassifier(strategy="prior").fit(["x"] * 2, [1, 5])
    model.class_prior_ = np.array([np.nan, np.nan])
    path = model_dir / "second.joblib"
    joblib.dump(model, path)
    manifest = json.loads((model_dir / "manifest.json").read_text())
    manifest["models"][1]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (model_dir / "manifest.json").write_text(json.dumps(manifest))
    with TestClient(api.create_app(model_dir=model_dir, history_db=db)) as client:
        assert (
            client.post("/load_model", json={"model_id": "second"}).status_code == 503
        )
        assert (
            client.post("/predict", json={"Review": "Still works"}).json()["label"] == 2
        )


def test_opt_in_raw_text_is_persisted(service_files):
    model_dir, db = service_files
    with TestClient(
        api.create_app(model_dir=model_dir, history_db=db, store_review_text=True)
    ) as client:
        assert (
            client.post("/predict", json={"Review": "Exact review"}).status_code == 200
        )
    with sqlite3.connect(db) as connection:
        assert (
            connection.execute("SELECT review_text FROM predictions").fetchone()[0]
            == "Exact review"
        )


def test_streaming_body_limit_and_malformed_json(client):
    response = client.post("/predict", content=(b" " * 600_000 for _ in range(2)))
    assert response.status_code == 413
    response = client.post(
        "/predict", content=b'{"Review":', headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["message"] == "Некорректный JSON"


def test_non_unicode_surrogate_is_a_validation_error(client):
    response = client.post(
        "/predict",
        content=b'{"Review":"\\ud800"}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_manifest_rejects_path_outside_model_directory(service_files):
    model_dir, db = service_files
    manifest = json.loads((model_dir / "manifest.json").read_text())
    manifest["models"][0]["filename"] = "../outside.joblib"
    (model_dir / "manifest.json").write_text(json.dumps(manifest))
    from company_reviews.model_registry import ModelLoadError

    with (
        pytest.raises(ModelLoadError),
        TestClient(api.create_app(model_dir=model_dir, history_db=db)),
    ):
        pass


def test_median_and_confidence_differ_from_most_likely_class(service_files):
    model_dir, db = service_files
    model = DummyClassifier(strategy="prior").fit(["x"] * 5, [1, 1, 3, 5, 5])
    path = model_dir / "first.joblib"
    joblib.dump(model, path)
    manifest = json.loads((model_dir / "manifest.json").read_text())
    manifest["models"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (model_dir / "manifest.json").write_text(json.dumps(manifest))
    with TestClient(api.create_app(model_dir=model_dir, history_db=db)) as client:
        response = client.post("/predict", json={"Review": "An ambiguous experience"})
    assert response.status_code == 200
    assert response.json() == {"label": 3, "confidence": 0.2}
