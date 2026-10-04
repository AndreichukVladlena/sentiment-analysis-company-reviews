#!/usr/bin/env python3
"""Check a running API and optionally measure warm HTTP latency using stdlib only."""

import argparse
import json
import math
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SHORT_REVIEW = (
    "Excellent service. My order arrived on time and the support team was helpful."
)
MEDIUM_REVIEW = (
    "I ordered a replacement appliance after comparing several shops. "
    "The website was easy to use, and the confirmation email arrived immediately. "
    "Delivery took one day longer than promised, but customer support explained the delay "
    "and provided a tracking update. The item was packed carefully and worked as expected. "
    "I would probably order again, although more accurate delivery estimates would help."
)
LONG_REVIEW = (
    "I have used this company several times over the last year, with mixed results. "
    "The product descriptions usually match what arrives, and the prices seem reasonable. "
    "For my most recent order, the payment went through immediately but I did not receive "
    "a dispatch notification until I contacted support two days later. The first reply "
    "was a generic message, while the second agent checked the order and explained that "
    "one item was temporarily unavailable. I agreed to a replacement and received the "
    "parcel the following week. Everything was securely packaged, although the outer "
    "box had a dent. The replacement worked well and the missing item was refunded. "
    "I appreciate that the issue was eventually resolved, but the communication required "
    "more effort than I expected. I would consider ordering again for something that "
    "is not urgent, provided the company improves its stock information and proactively "
    "notifies customers about delays. A clear delivery estimate and a useful update "
    "would have saved several emails and made the experience much less frustrating. "
    "The support staff remained polite throughout, and I did not have to repeat my "
    "payment details or provide unnecessary documents. Overall, the product itself "
    "met my expectations, while the delivery process and communication could be better."
)
REVIEWS = {"short": SHORT_REVIEW, "medium": MEDIUM_REVIEW, "long": LONG_REVIEW}


class SmokeFailure(RuntimeError):
    """The running service did not meet the expected HTTP contract."""


def require(condition, message):
    if not condition:
        raise SmokeFailure(message)


class APIClient:
    def __init__(self, base_url, timeout):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, path, payload=None, expected_status=200, json_response=True):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        started = perf_counter()
        try:
            response = urlopen(request, timeout=self.timeout)
        except HTTPError as exc:
            response = exc
        with response:
            content = response.read()
            status = response.status
            headers = response.headers
        elapsed_ms = (perf_counter() - started) * 1000
        require(
            status == expected_status,
            f"{request.get_method()} {path}: expected HTTP {expected_status}, got {status}: "
            f"{content[:300].decode('utf-8', errors='replace')}",
        )
        if json_response:
            try:
                content = json.loads(content)
            except (ValueError, UnicodeError) as exc:
                raise SmokeFailure(f"{path}: response is not valid JSON") from exc
        return content, headers, elapsed_ms


def check_prediction(prediction):
    require(isinstance(prediction, dict), "Prediction must be an object")
    require(set(prediction) == {"label", "confidence"}, "Unexpected prediction fields")
    label = prediction["label"]
    confidence = prediction["confidence"]
    require(type(label) is int and 1 <= label <= 5, "label must be an integer in 1–5")
    require(
        type(confidence) in (int, float)
        and math.isfinite(confidence)
        and 0 <= confidence <= 1,
        "confidence must be a finite number in 0–1",
    )


def predict(client, rows, model_id=None):
    data, headers, elapsed_ms = client.request("/predict", rows)
    if isinstance(rows, list):
        require(
            isinstance(data, list) and len(data) == len(rows), "Batch shape mismatch"
        )
        for prediction in data:
            check_prediction(prediction)
    else:
        check_prediction(data)
    require(bool(headers.get("X-Request-Id")), "Missing X-Request-Id history reference")
    require(bool(headers.get("X-Model-Version")), "Missing X-Model-Version")
    if model_id is not None:
        require(
            headers.get("X-Model-Id") == model_id,
            f"Expected model {model_id} in response",
        )
    return elapsed_ms


def select_model(client, model_id):
    data, _, _ = client.request("/load_model", {"model_id": model_id})
    require(data.get("model_id") == model_id, "Model selection response mismatch")


def smoke(client):
    health, _, _ = client.request("/health")
    require(health.get("status") == "ok", "Service is not ready")
    docs, _, _ = client.request("/docs", json_response=False)
    require(b"swagger" in docs.lower(), "Swagger HTML is missing")
    schema, _, _ = client.request("/openapi.json")
    for endpoint in ("/predict", "/load_model"):
        require(
            "post" in schema.get("paths", {}).get(endpoint, {}),
            f"Missing {endpoint} OpenAPI operation",
        )
    catalog, _, _ = client.request("/models")
    model_ids = {model["id"] for model in catalog.get("models", [])}
    require(
        {"word-only", "tfidf"} <= model_ids, "Expected word-only and tfidf in catalog"
    )

    predict(client, {"Id": 17, "Review": SHORT_REVIEW})
    predict(
        client,
        [
            {"Review": SHORT_REVIEW},
            {"Id": 18, "Review": "Terrible service. My order never arrived."},
        ],
    )
    client.request("/predict", {"Review": "   "}, expected_status=422)
    client.request("/predict", {"Review": 42}, expected_status=422)

    for model_id in ("word-only", "tfidf"):
        select_model(client, model_id)
        predict(client, {"Review": SHORT_REVIEW}, model_id)
        client.request(
            "/load_model", {"model_id": "smoke-unknown-model"}, expected_status=404
        )
        health, _, _ = client.request("/health")
        require(
            health.get("model_id") == model_id, "Failed selection changed active model"
        )
        predict(client, {"Review": MEDIUM_REVIEW}, model_id)
    print(
        "Smoke passed: health, Swagger, prediction shapes, validation, model selection and failed selection recovery."
    )


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values):
    return {
        "requests": len(values),
        "p50_ms": round(percentile(values, 0.50), 3),
        "p95_ms": round(percentile(values, 0.95), 3),
        "max_ms": round(max(values), 3),
    }


def benchmark(client, environment):
    select_model(client, "tfidf")
    singles = {}
    for name, review in REVIEWS.items():
        require(len(review) <= 2000, "Benchmark review exceeds 2,000 characters")
        payload = {"Review": review}
        for _ in range(3):
            predict(client, payload, "tfidf")
        timings = [predict(client, payload, "tfidf") for _ in range(30)]
        singles[name] = {"characters": len(review), **summarize(timings)}

    texts = list(REVIEWS.values())
    batch = [{"Review": texts[index % len(texts)]} for index in range(16)]
    for _ in range(3):
        predict(client, batch, "tfidf")
    batch_timings = [predict(client, batch, "tfidf") for _ in range(30)]
    health, _, _ = client.request("/health")
    return {
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "base_url": client.base_url,
        "model_id": health["model_id"],
        "model_version": health["model_version"],
        "environment_description": environment
        or "Not supplied; server hardware is unknown.",
        "client_platform": platform.platform(),
        "client_python": platform.python_version(),
        "measurement": (
            "Sequential warm HTTP requests including server inference and synchronous SQLite history writes; "
            "client wall-clock latency from urlopen through reading the response. "
            "No concurrent load. Synthetic English reviews, not dataset samples. "
            "Model startup and three warm-up requests per case are excluded. "
            "Percentiles use linear interpolation. Client platform does not describe remote server hardware."
        ),
        "single": singles,
        "batch16": {
            "items": 16,
            "characters_total": sum(len(row["Review"]) for row in batch),
            **summarize(batch_timings),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000", help="Running API URL"
    )
    parser.add_argument(
        "--timeout", type=float, default=15, help="Timeout per request in seconds"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Measure 30 requests per text length and 30 batches of 16",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write benchmark JSON to this file (requires --benchmark)",
    )
    parser.add_argument(
        "--environment",
        help="Explicit server hardware/runtime description supplied by the operator",
    )
    args = parser.parse_args()
    if args.output and not args.benchmark:
        parser.error("--output requires --benchmark")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    client = APIClient(args.base_url, args.timeout)
    try:
        try:
            smoke(client)
            if args.benchmark:
                report = benchmark(client, args.environment)
                rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(rendered, encoding="utf-8")
                    print(f"Benchmark saved to {args.output}")
                print(rendered, end="")
        finally:
            # Leave the default serving model selected even after a smoke failure.
            select_model(client, "tfidf")
    except (SmokeFailure, URLError, TimeoutError, OSError) as exc:
        print(f"Smoke failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
