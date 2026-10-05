"""Check a running service: python scripts/smoke_api.py [http://localhost:8000]."""

import json
import sys
from urllib.request import Request, urlopen

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def request(path, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    req = Request(
        BASE_URL + path,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=10) as response:
        return json.load(response)


def check_prediction(prediction):
    assert type(prediction["label"]) is int
    assert 1 <= prediction["label"] <= 5
    assert 0 <= prediction["confidence"] <= 1


def main():
    original = request("/health")["model_id"]
    try:
        request("/load_model", {"model_id": "tfidf"})
        check_prediction(request("/predict", {"Review": "Excellent service!"}))

        predictions = request(
            "/predict",
            [
                {"Id": 1, "Review": "Excellent service!"},
                {"Id": 2, "Review": "My order never arrived."},
            ],
        )
        assert len(predictions) == 2
        for prediction in predictions:
            check_prediction(prediction)

        request("/load_model", {"model_id": "word-only"})
        assert request("/health")["model_id"] == "word-only"
        check_prediction(request("/predict", {"Review": "Helpful staff."}))

        with urlopen(BASE_URL + "/docs", timeout=10) as response:
            assert response.status == 200
        print("OK: single/batch predictions, model switching and Swagger")
    finally:
        request("/load_model", {"model_id": original})


if __name__ == "__main__":
    main()
