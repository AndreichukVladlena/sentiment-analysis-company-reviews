"""Reproducible, sequential TF-IDF training and grouped model selection.

Run from the repository root: python -m company_reviews.training --tune
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

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

    def __init__(
        self,
        min_df=5,
        char_min_df=5,
        char_weight=0.5,
        numeric_weight=0.1,
        numeric_columns=None,
    ):
        self.min_df = min_df
        self.char_min_df = char_min_df
        self.char_weight = char_weight
        self.numeric_weight = numeric_weight
        self.numeric_columns = numeric_columns

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
            self.scaler_ = StandardScaler().fit(self._numeric(text))

        return self

    def _numeric(self, text):
        values = numeric_feature_matrix(text)
        if self.numeric_columns is not None:
            values = values[:, self.numeric_columns]
        return values

    def transform(self, X):
        check_is_fitted(self, "word_vectorizer_")
        text = pd.Series(X, dtype="str").reset_index(drop=True)
        word = self.word_vectorizer_.transform(text)

        char = None
        if self.char_vectorizer_ is not None:
            char = self.char_vectorizer_.transform(text)

        numeric = None
        if self.scaler_ is not None:
            numeric = self.scaler_.transform(self._numeric(text))

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
    features = ReviewFeatures(
        min_df=min_df,
        char_weight=0 if word_only else 0.5,
        numeric_weight=0 if word_only else 0.1,
    )
    classifier = LogisticRegression(C=c, max_iter=250, solver="lbfgs")

    return Pipeline(
        [("features", features), ("classifier", classifier)],
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
    numeric_columns = ()
    if features.numeric_weight:
        numeric_columns = tuple(range(13))
        if features.numeric_columns is not None:
            numeric_columns = tuple(features.numeric_columns)

    return BaselineModel(
        word_vectorizer=features.word_vectorizer_,
        char_vectorizer=features.char_vectorizer_,
        scaler=features.scaler_,
        classifier=pipeline.named_steps["classifier"],
        char_weight=features.char_weight,
        numeric_weight=features.numeric_weight,
        numeric_columns=numeric_columns,
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


def save_models(full_model, word_model, output=Path("models")):
    """Save two trusted models and the catalog used by /load_model."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model_version = datetime.now(UTC).isoformat(timespec="seconds")

    def save(model_id, pipeline, description):
        path = output / f"{model_id}.joblib"
        joblib.dump(export_model(pipeline), path, compress=3)
        return {
            "id": model_id,
            "filename": path.name,
            "version": model_version,
            "description": description,
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "sklearn_version": version("scikit-learn"),
        }

    full = save(
        "tfidf",
        full_model,
        "TF-IDF по словам и символам + 13 числовых признаков + Logistic Regression",
    )
    word = save("word-only", word_model, "Word and bigram TF-IDF")
    catalog = {"default_model": "tfidf", "models": [full, word]}
    (output / "manifest.json").write_text(json.dumps(catalog, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument(
        "--tune", action="store_true", help="Run the 9-combination grid search"
    )
    args = parser.parse_args()

    data = read_training_data(args.data)
    reviews = data["Review"]
    labels = data["Rating"]
    groups = normalize_reviews_for_split(reviews)

    # Cache only Pipeline transformations, not completed searches or scores.
    memory = joblib.Memory("data/cache/tfidf-transforms", verbose=0)

    with threadpool_limits(limits=2):
        if args.tune:
            search = build_search(memory=memory)
            search.fit(reviews, labels, groups=groups)
            full_model = search.best_estimator_
            print(f"Parameters: {search.best_params_}")
            print(f"Mean CV MAE: {-search.best_score_:.5f}")
        else:
            full_model = build_pipeline()
            full_model.fit(reviews, labels)

        word_model = build_pipeline(word_only=True)
        word_model.fit(reviews, labels)
        save_models(full_model, word_model, args.output)

    print(f"Saved tfidf.joblib, word-only.joblib and manifest.json to {args.output}")


if __name__ == "__main__":
    main()
