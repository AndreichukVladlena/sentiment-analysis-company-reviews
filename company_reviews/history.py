"""Atomic SQLite history; review text is excluded unless explicitly enabled."""

import hashlib
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from company_reviews.model_registry import ModelSpec


class HistoryStore:
    def __init__(self, path: Path, store_review_text: bool = False):
        self.path = path
        self.store_review_text = store_review_text
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    model_sha256 TEXT NOT NULL,
                    item_count INTEGER NOT NULL,
                    inference_ms REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    request_id TEXT NOT NULL REFERENCES requests(request_id),
                    item_index INTEGER NOT NULL,
                    dataset_id INTEGER,
                    review_sha256 TEXT NOT NULL,
                    review_length INTEGER NOT NULL,
                    review_text TEXT,
                    label INTEGER NOT NULL CHECK (label BETWEEN 1 AND 5),
                    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
                    PRIMARY KEY (request_id, item_index)
                );
            """)

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def record(
        self,
        rows: Sequence[tuple[int | None, str]],
        predictions: Sequence[dict],
        model: ModelSpec,
        inference_ms: float,
    ) -> str:
        request_id = uuid4().hex
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    "INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        request_id,
                        datetime.now(UTC).isoformat(),
                        model.id,
                        model.version,
                        model.sha256,
                        len(rows),
                        inference_ms,
                    ),
                )
                connection.executemany(
                    "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            request_id,
                            index,
                            dataset_id,
                            hashlib.sha256(text.encode("utf-8")).hexdigest(),
                            len(text),
                            text if self.store_review_text else None,
                            prediction["label"],
                            prediction["confidence"],
                        )
                        for index, ((dataset_id, text), prediction) in enumerate(
                            zip(rows, predictions, strict=True)
                        )
                    ],
                )
        finally:
            connection.close()
        return request_id
