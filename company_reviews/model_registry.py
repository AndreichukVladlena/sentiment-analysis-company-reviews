"""Trusted local model catalog and atomic, process-local model selection."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import file_digest, sha256
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal

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
    format: Literal["joblib", "deberta"] = "joblib"
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    sklearn_version: str | None = None
    file_sha256: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def required_checksums(self):
        if self.format == "joblib" and (not self.sha256 or not self.sklearn_version):
            raise ValueError("Joblib requires sha256 and sklearn_version")
        if self.format == "deberta" and not self.file_sha256:
            raise ValueError("DeBERTa requires per-file checksums")
        if self.format == "deberta":
            # History stores a single fingerprint of all files, including the tokenizer.
            self.sha256 = sha256(
                json.dumps(self.file_sha256, sort_keys=True).encode("utf-8")
            ).hexdigest()
        return self


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

    def _check_deberta_files(self, path: Path, spec: ModelSpec) -> None:
        required = {
            "config.json",
            "model.safetensors",
            "tokenizer_config.json",
            "spm.model",
        }
        allowed = required | {
            "special_tokens_map.json",
            "added_tokens.json",
            "training_config.json",
        }
        if not required <= spec.file_sha256.keys() <= allowed:
            raise ModelLoadError("DeBERTa manifest has missing or unsupported files")
        if {file.name for file in path.iterdir()} != spec.file_sha256.keys():
            raise ModelLoadError("DeBERTa directory differs from its manifest")
        for name, checksum in spec.file_sha256.items():
            file = (path / name).resolve()
            if file.parent != path or not file.is_file():
                raise ModelLoadError("DeBERTa files must stay inside their directory")
            # Check the hash without keeping another copy of the weights (~740 MB) in memory.
            with file.open("rb") as source:
                if file_digest(source, "sha256").hexdigest() != checksum:
                    raise ModelLoadError("DeBERTa file checksum mismatch")

    def load(self, model_id: str) -> LoadedModel:
        if model_id not in self._specs:
            raise UnknownModelError(model_id)
        spec = self._specs[model_id]
        # Serialize loads, but let requests continue using the previous model.
        with self._load_lock:
            try:
                path = (self.model_dir / spec.filename).resolve()
                if path.parent != self.model_dir:
                    raise ModelLoadError(
                        "Model file must be directly inside model directory"
                    )
                if spec.format == "deberta":
                    self._check_deberta_files(path, spec)
                    # Selecting the same model again does not create another copy in memory.
                    if self._active is not None and self._active.spec.id == model_id:
                        return self._active
                    from company_reviews.deberta import DebertaClassifier

                    estimator = DebertaClassifier(path)
                else:
                    if path.suffix != ".joblib":
                        raise ModelLoadError("Expected a joblib model file")
                    if spec.sklearn_version != sklearn.__version__:
                        raise ModelLoadError(
                            "Training and serving sklearn versions differ"
                        )
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
