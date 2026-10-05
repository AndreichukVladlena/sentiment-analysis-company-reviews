"""Подготовка признаков и GridSearchCV для baselinev2.ipynb."""

from __future__ import annotations

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

from company_reviews.baseline import (
    BaselineModel,
    combine_features,
    median_from_probabilities,
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
