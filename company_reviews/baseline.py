"""Признаки и прогноз TF-IDF baseline для отзывов компаний."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURE_NAMES = (
    "log_chars",
    "log_words",
    "has_exclamation",
    "has_question",
    "log_uppercase_words",
    "log_exclamation_count",
    "log_question_count",
    "mentions_0",
    "mentions_1",
    "mentions_2",
    "mentions_3",
    "mentions_4",
    "mentions_5",
)


def normalize_reviews_for_split(reviews: pd.Series) -> pd.Series:
    """Убрать различия регистра и пробелов при поиске совпадающих отзывов."""
    return reviews.str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()


def median_from_probabilities(
    probabilities: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Выбрать медиану распределения оценок в каждой строке."""
    return labels[np.argmax(np.cumsum(probabilities, axis=1) >= 0.5, axis=1)]


def feature_groups(reviews: pd.Series) -> dict[str, np.ndarray]:
    """Вернуть группы числовых признаков в фиксированном порядке."""
    text = reviews.fillna("")

    eda = np.column_stack(
        [
            np.log1p(text.str.len().to_numpy()),
            np.log1p(text.str.split().str.len().to_numpy()),
            text.str.contains("!", regex=False).to_numpy(dtype=np.float32),
            text.str.contains("?", regex=False).to_numpy(dtype=np.float32),
        ]
    )

    uppercase = np.log1p(text.str.count(r"\b[A-Z]{3,}\b").to_numpy()).reshape(-1, 1)

    punctuation_counts = np.column_stack(
        [
            np.log1p(text.str.count("!").to_numpy()),
            np.log1p(text.str.count(r"\?").to_numpy()),
        ]
    )

    mentioned = (
        text.str.extract(
            r"(?i)(?<!\w)([0-5]|zero|one|two|three|four|five)\s*"
            r"(?:-\s*)?stars?\b|\b([0-5])\s*/\s*5\b"
        )
        .bfill(axis=1)
        .iloc[:, 0]
        .str.lower()
    )
    score_words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }
    mentioned = mentioned.replace(score_words).fillna(-1).astype(int).to_numpy()
    mentioned_flags = (mentioned[:, None] == np.arange(6)).astype(np.float32)

    return {
        "EDA": eda.astype(np.float32),
        "прописные слова": uppercase.astype(np.float32),
        "число ! и ?": punctuation_counts.astype(np.float32),
        "упоминание оценки": mentioned_flags,
    }


def numeric_feature_matrix(reviews: pd.Series) -> np.ndarray:
    """Собрать 13 числовых признаков в порядке NUMERIC_FEATURE_NAMES."""
    return np.column_stack(list(feature_groups(reviews).values()))


def combine_features(
    word_matrix: spmatrix,
    char_matrix: spmatrix | None,
    numeric_matrix: np.ndarray | None,
    char_weight: float,
    numeric_weight: float,
) -> csr_matrix:
    """Соединить TF-IDF по словам, символам и числовые признаки."""
    parts = [word_matrix]

    if char_matrix is not None and char_weight:
        parts.append(char_weight * char_matrix)

    if numeric_matrix is not None and numeric_weight:
        parts.append(csr_matrix(numeric_weight * numeric_matrix))

    combined = hstack(parts, format="csr")
    combined.sort_indices()
    return combined


@dataclass
class BaselineModel:
    """Обученные преобразования и классификатор с единым predict."""

    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer | None
    scaler: StandardScaler | None
    classifier: LogisticRegression
    char_weight: float
    numeric_weight: float
    numeric_columns: tuple[int, ...] = tuple(range(13))

    def _matrix(self, reviews: pd.Series | Sequence[str]) -> csr_matrix:
        text = pd.Series(reviews, dtype="str").reset_index(drop=True)
        word_matrix = self.word_vectorizer.transform(text)

        char_matrix = None
        if self.char_vectorizer is not None:
            char_matrix = self.char_vectorizer.transform(text)

        numeric_matrix = None
        if self.scaler is not None and self.numeric_columns:
            raw_numeric = numeric_feature_matrix(text)[:, self.numeric_columns]
            numeric_matrix = self.scaler.transform(raw_numeric)

        return combine_features(
            word_matrix,
            char_matrix,
            numeric_matrix,
            self.char_weight,
            self.numeric_weight,
        )

    def predict_proba(self, reviews: pd.Series | Sequence[str]) -> np.ndarray:
        return self.classifier.predict_proba(self._matrix(reviews))

    def predict(self, reviews: pd.Series | Sequence[str]) -> np.ndarray:
        probabilities = self.predict_proba(reviews)
        return median_from_probabilities(probabilities, self.classifier.classes_)
