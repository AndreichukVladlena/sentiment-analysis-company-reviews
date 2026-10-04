"""Reproducible, sequential TF-IDF training and grouped model selection.

Run from the repository root: python -m company_reviews.training --tune
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from company_reviews.baseline import (
    BaselineModel,
    combine_features,
    median_from_probabilities,
    normalize_reviews_for_split,
    numeric_feature_matrix,
)


class ReviewFeatures(TransformerMixin, BaseEstimator):
    """Fit all learned text and numeric preprocessing inside each CV fold."""

    def __init__(self, min_df=5, char_min_df=5, char_weight=0.5, numeric_weight=0.1):
        self.min_df = min_df
        self.char_min_df = char_min_df
        self.char_weight = char_weight
        self.numeric_weight = numeric_weight

    def fit(self, X, y=None):
        text = pd.Series(X, dtype="str").reset_index(drop=True)
        self.word_vectorizer_ = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=self.min_df,
            max_features=150_000,
            sublinear_tf=True,
            dtype=np.float32,
        ).fit(text)

        self.char_vectorizer_ = None
        if self.char_weight:
            self.char_vectorizer_ = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=self.char_min_df,
                max_features=100_000,
                sublinear_tf=True,
                dtype=np.float32,
            ).fit(text)

        self.scaler_ = None
        if self.numeric_weight:
            self.scaler_ = StandardScaler().fit(numeric_feature_matrix(text))

        return self

    def transform(self, X):
        check_is_fitted(self, "word_vectorizer_")
        text = pd.Series(X, dtype="str").reset_index(drop=True)
        word = self.word_vectorizer_.transform(text)
        char = self.char_vectorizer_.transform(text) if self.char_vectorizer_ else None
        numeric = (
            self.scaler_.transform(numeric_feature_matrix(text))
            if self.scaler_ is not None
            else None
        )
        return combine_features(
            word, char, numeric, self.char_weight, self.numeric_weight
        )


def negative_median_mae(estimator, reviews, labels):
    """GridSearch maximizes scores; use minus MAE of the probability median."""
    prediction = median_from_probabilities(
        estimator.predict_proba(reviews), estimator.classes_
    )
    return -mean_absolute_error(labels, prediction)


def build_pipeline(*, min_df=5, c=4, word_only=False, memory=None):
    return Pipeline(
        [
            (
                "features",
                ReviewFeatures(
                    min_df=min_df,
                    char_weight=0 if word_only else 0.5,
                    numeric_weight=0 if word_only else 0.1,
                ),
            ),
            ("classifier", LogisticRegression(C=c, max_iter=250, solver="lbfgs")),
        ],
        memory=memory,
    )


def build_search(
    *, memory=None, n_splits=5, c_values=(2, 4, 8), min_df_values=(2, 3, 5), verbose=2
):
    return GridSearchCV(
        build_pipeline(memory=memory),
        {"classifier__C": list(c_values), "features__min_df": list(min_df_values)},
        scoring=negative_median_mae,
        cv=StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=2026),
        n_jobs=1,
        pre_dispatch=1,
        refit=True,
        error_score="raise",
        verbose=verbose,
    )


def export_model(pipeline):
    features = pipeline.named_steps["features"]
    return BaselineModel(
        features.word_vectorizer_,
        features.char_vectorizer_,
        features.scaler_,
        pipeline.named_steps["classifier"],
        features.char_weight,
        features.numeric_weight,
        tuple(range(13)) if features.numeric_weight else (),
    )


def read_training_data(path):
    data = pd.read_csv(path)
    if list(data.columns) != ["Id", "Review", "Rating"]:
        raise ValueError("Expected columns: Id, Review, Rating")
    if (
        data["Review"].isna().any()
        or not data["Review"].map(lambda x: isinstance(x, str)).all()
    ):
        raise ValueError("Review must contain non-null strings")

    if not data["Rating"].isin(range(1, 6)).all():
        raise ValueError("Rating must be an integer from 1 through 5")

    if data.groupby("Review")["Rating"].nunique().max() > 1:
        raise ValueError("Exact duplicate reviews have conflicting ratings")

    return data.drop_duplicates("Review", keep="first").reset_index(drop=True)


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_fingerprint(data_path, configuration):
    payload = {
        "data_sha256": file_sha256(data_path),
        "code_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("baseline.py", "training.py")
        },
        "versions": {
            name: version(name)
            for name in ("numpy", "scipy", "pandas", "scikit-learn", "joblib")
        },
        "configuration": configuration,
    }
    key = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return key, payload


def train(
    *,
    data_path=Path("data/raw/train.csv"),
    output=Path("models"),
    cache=Path("data/cache/grid-search"),
    tune=False,
    c=4,
    min_df=5,
    threads=2,
):
    """Train two deployment artifacts; optional grid search caches fold transforms."""
    data_path, output, cache = map(Path, (data_path, output, cache))
    data = read_training_data(data_path)
    reviews, labels = data["Review"], data["Rating"].to_numpy()
    groups = normalize_reviews_for_split(reviews)

    configuration = {
        "tune": tune,
        "C": c,
        "min_df": min_df,
        "threads": threads,
        "seed": 2026,
        "c_grid": [2, 4, 8],
        "min_df_grid": [2, 3, 5],
        "char_weight": 0.5,
        "numeric_weight": 0.1,
    }

    key, provenance = training_fingerprint(data_path, configuration)
    run_cache = cache / key
    run_cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    metrics = {
        "provenance": provenance,
        "training_id": key,
        "rows": len(data),
        "groups": int(groups.nunique()),
        "evaluation_note": "Grouped 5-fold CV was used for feature and parameter selection; "
        "these scores are not an untouched holdout estimate.",
    }

    with threadpool_limits(limits=threads):
        memory = joblib.Memory(run_cache / "transforms", verbose=0)
        if tune:
            search_path = run_cache / "search.joblib"
            if search_path.exists():
                search = joblib.load(search_path)
                print(f"Loaded completed search {key}", flush=True)
            else:
                search = build_search(memory=memory)
                search.fit(reviews, labels, groups=groups)
                joblib.dump(search, search_path, compress=3)

            selected = search.best_estimator_
            c = search.best_params_["classifier__C"]
            min_df = search.best_params_["features__min_df"]

            results = pd.DataFrame(search.cv_results_)
            columns = [
                "param_classifier__C",
                "param_features__min_df",
                "mean_test_score",
                "std_test_score",
                "rank_test_score",
            ] + [f"split{i}_test_score" for i in range(5)]
            table = results[columns].copy()
            for column in table.columns:
                if (
                    column.endswith("test_score")
                    and column != "rank_test_score"
                    and not column.startswith("std_")
                ):
                    table[column] *= -1
            table.to_csv(output / "grid_search.csv", index=False)
            metrics["selected_cv_mae"] = -float(search.best_score_)
            metrics["selected_fold_mae"] = [
                -float(results.iloc[search.best_index_][f"split{i}_test_score"])
                for i in range(5)
            ]
        else:
            selected = build_pipeline(min_df=min_df, c=c).fit(reviews, labels)
        metrics["selected_parameters"] = {
            "C": c,
            "min_df": min_df,
            "char_weight": 0.5,
            "numeric_weight": 0.1,
        }

        evaluation_path = run_cache / "evaluation.json"
        if evaluation_path.exists():
            evaluation = json.loads(evaluation_path.read_text())
        else:
            folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
            word_scores, dummy_scores = [], []
            evaluation = {}
            for fold, (fit_idx, valid_idx) in enumerate(
                folds.split(reviews, labels, groups)
            ):
                assert set(groups.iloc[fit_idx]).isdisjoint(groups.iloc[valid_idx])
                word = build_pipeline(min_df=5, c=4, word_only=True).fit(
                    reviews.iloc[fit_idx], labels[fit_idx]
                )
                word_scores.append(
                    -negative_median_mae(
                        word, reviews.iloc[valid_idx], labels[valid_idx]
                    )
                )
                dummy_scores.append(
                    float(np.abs(labels[valid_idx] - np.median(labels[fit_idx])).mean())
                )
                if fold == 0:
                    full = build_pipeline(min_df=min_df, c=c, memory=memory).fit(
                        reviews.iloc[fit_idx], labels[fit_idx]
                    )
                    evaluation["first_fold"] = {
                        "rows": len(valid_idx),
                        "training_rows": len(fit_idx),
                        "split_sha256": sha256(
                            fit_idx.astype("<i8").tobytes()
                            + valid_idx.astype("<i8").tobytes()
                        ).hexdigest(),
                        "selected_mae": -negative_median_mae(
                            full, reviews.iloc[valid_idx], labels[valid_idx]
                        ),
                        "word_only_mae": word_scores[-1],
                        "dummy_mae": dummy_scores[-1],
                    }
                print(
                    f"Baseline evaluation fold {fold + 1}/5: {word_scores[-1]:.5f}",
                    flush=True,
                )
            evaluation.update(
                word_only_cv_mae=float(np.mean(word_scores)),
                word_only_fold_mae=word_scores,
                dummy_cv_mae=float(np.mean(dummy_scores)),
                dummy_fold_mae=dummy_scores,
            )
            evaluation_path.write_text(json.dumps(evaluation, indent=2) + "\n")

        metrics.update(evaluation)
        word = build_pipeline(min_df=5, c=4, word_only=True).fit(reviews, labels)
        entries = []
        for model_id, pipeline, description in (
            (
                "tfidf",
                selected,
                "Word and character TF-IDF with 13 numeric features; probability median",
            ),
            ("word-only", word, "Word TF-IDF baseline; probability median"),
        ):
            path = output / f"{model_id}.joblib"
            model = export_model(pipeline)
            joblib.dump(model, path, compress=3)
            restored = joblib.load(path)
            np.testing.assert_allclose(
                restored.predict_proba(reviews.head(10)),
                model.predict_proba(reviews.head(10)),
            )
            entries.append(
                {
                    "id": model_id,
                    "filename": path.name,
                    "version": key,
                    "description": description,
                    "sha256": file_sha256(path),
                    "sklearn_version": version("scikit-learn"),
                    "bytes": path.stat().st_size,
                }
            )
        (output / "manifest.json").write_text(
            json.dumps({"default_model": "tfidf", "models": entries}, indent=2) + "\n"
        )
    metrics["training_seconds"] = perf_counter() - started
    (output / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache/grid-search"))
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run 9 parameter combinations over 5 grouped folds",
    )
    parser.add_argument("--c", type=float, default=4)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or args.c <= 0 or args.min_df < 1:
        parser.error("threads and min-df must be >= 1; c must be positive")
    train(
        data_path=args.data,
        output=args.output,
        cache=args.cache,
        tune=args.tune,
        c=args.c,
        min_df=args.min_df,
        threads=args.threads,
    )


if __name__ == "__main__":
    # Persist cached sklearn objects under their importable module, not __main__.
    from company_reviews.training import main as run_main

    run_main()
