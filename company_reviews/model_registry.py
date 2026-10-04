"""Trusted local model catalog and atomic, process-local model selection."""

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import numpy as np
import sklearn
from pydantic import BaseModel, Field, model_validator


class ModelLoadError(RuntimeError):
    """A registered model cannot safely serve predictions."""


class UnknownModelError(KeyError):
    """The requested ID is not in the trusted catalog."""


class ModelSpec(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    filename: str
    version: str = Field(min_length=1, max_length=128)
    description: str = ""
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sklearn_version: str


class Manifest(BaseModel):
    default_model: str
    models: list[ModelSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_catalog(self):
        ids = [model.id for model in self.models]
        if len(ids) != len(set(ids)) or self.default_model not in ids:
            raise ValueError(
                "Manifest requires unique IDs and a registered default_model"
            )
        return self


@dataclass(frozen=True)
class LoadedModel:
    spec: ModelSpec
    estimator: Any
    classes: np.ndarray

    def predict(self, reviews: Sequence[str]) -> list[dict[str, int | float]]:
        """Use the posterior median for MAE; confidence refers to that label."""
        probabilities = np.asarray(self.estimator.predict_proba(reviews), dtype=float)
        if (
            probabilities.shape != (len(reviews), len(self.classes))
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)
        ):
            raise ModelLoadError("Model returned invalid probabilities")
        indices = np.argmax(np.cumsum(probabilities, axis=1) >= 0.5, axis=1)
        return [
            {"label": int(self.classes[index]), "confidence": float(row[index])}
            for row, index in zip(probabilities, indices, strict=True)
        ]


class ModelRegistry:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir.resolve()
        self.manifest = Manifest.model_validate_json(
            (self.model_dir / "manifest.json").read_text(encoding="utf-8")
        )
        self._specs = {spec.id: spec for spec in self.manifest.models}
        self._swap_lock = Lock()
        self._load_lock = Lock()
        self._active: LoadedModel | None = None
        self.load(self.manifest.default_model)

    def snapshot(self) -> LoadedModel:
        with self._swap_lock:
            if self._active is None:
                raise ModelLoadError("No active model")
            return self._active

    def load(self, model_id: str) -> LoadedModel:
        if model_id not in self._specs:
            raise UnknownModelError(model_id)
        spec = self._specs[model_id]
        # Serialize loads, but let requests continue using the previous model.
        with self._load_lock:
            try:
                path = (self.model_dir / spec.filename).resolve()
                if path.parent != self.model_dir or path.suffix != ".joblib":
                    raise ModelLoadError(
                        "Model file must be directly inside model directory"
                    )
                if spec.sklearn_version != sklearn.__version__:
                    raise ModelLoadError("Training and serving sklearn versions differ")
                contents = path.read_bytes()
                if sha256(contents).hexdigest() != spec.sha256:
                    raise ModelLoadError("Model checksum mismatch")
                # Read once: the bytes whose checksum was checked are deserialized.
                estimator = joblib.load(BytesIO(contents))
                classes = getattr(estimator, "classes_", None)
                if classes is None:
                    classes = getattr(
                        getattr(estimator, "classifier", None), "classes_", None
                    )
                classes = np.asarray(classes)
                if (
                    classes.ndim != 1
                    or not len(classes)
                    or not np.isin(classes, [1, 2, 3, 4, 5]).all()
                    or not np.all(np.diff(classes.astype(int)) > 0)
                ):
                    raise ModelLoadError(
                        "Model classes must be sorted unique ratings 1–5"
                    )
                candidate = LoadedModel(spec, estimator, classes)
                candidate.predict(["A review for the model readiness check."])
            except Exception as exc:
                if isinstance(exc, ModelLoadError):
                    raise
                raise ModelLoadError("Model loading or readiness check failed") from exc
            with self._swap_lock:
                self._active = candidate
            return candidate
