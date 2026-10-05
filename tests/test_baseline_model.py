import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from company_reviews.baseline import (
    BaselineModel,
    combine_features,
    median_from_probabilities,
    normalize_reviews_for_split,
    numeric_feature_matrix,
)


class BaselineFeaturesTest(unittest.TestCase):
    def test_split_normalization_ignores_only_case_and_whitespace(self):
        reviews = pd.Series(["Great  Company ", "great company", "Bad company"])

        self.assertEqual(
            normalize_reviews_for_split(reviews).tolist(),
            ["great company", "great company", "bad company"],
        )

    def test_numeric_features_include_explicit_rating_mention(self):
        features = numeric_feature_matrix(pd.Series(["WOW!! one star?"]))

        self.assertEqual(features.shape, (1, 13))
        self.assertEqual(features[0, 2:4].tolist(), [1.0, 1.0])
        self.assertEqual(features[0, 7:13].tolist(), [0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    def test_feature_blocks_keep_word_char_numeric_order(self):
        word = csr_matrix([[1.0, 2.0], [0.0, 3.0]])
        char = csr_matrix([[4.0], [5.0]])
        numeric = np.array([[1.0], [2.0]], dtype=np.float32)

        result = combine_features(word, char, numeric, 0.5, 0.1)

        np.testing.assert_allclose(
            result.toarray(),
            [[1.0, 2.0, 2.0, 0.1], [0.0, 3.0, 2.5, 0.2]],
        )

    def test_probability_median_is_used_for_prediction(self):
        probabilities = np.array([[0.40, 0.20, 0.40], [0.10, 0.20, 0.70]])

        np.testing.assert_array_equal(
            median_from_probabilities(probabilities, np.array([1, 3, 5])),
            [3, 5],
        )


class BaselineModelTest(unittest.TestCase):
    def test_serialized_model_reproduces_manual_prediction(self):
        reviews = pd.Series(
            [
                "awful one star",
                "great five stars",
                "okay two stars",
                "excellent five stars",
                "bad one star",
                "fine three stars",
            ]
        )
        labels = np.array([1, 5, 2, 5, 1, 3])
        word = TfidfVectorizer(min_df=1).fit(reviews)
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(reviews)
        scaler = StandardScaler().fit(numeric_feature_matrix(reviews))
        matrix = combine_features(
            word.transform(reviews),
            char.transform(reviews),
            scaler.transform(numeric_feature_matrix(reviews)),
            char_weight=0.5,
            numeric_weight=0.1,
        )
        classifier = LogisticRegression(max_iter=100).fit(matrix, labels)
        model = BaselineModel(word, char, scaler, classifier, 0.5, 0.1)

        manual = classifier.predict_proba(matrix)
        np.testing.assert_allclose(model.predict_proba(reviews), manual)
        np.testing.assert_array_equal(
            model.predict(reviews),
            median_from_probabilities(manual, classifier.classes_),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(
                restored.predict(reviews), model.predict(reviews)
            )


if __name__ == "__main__":
    unittest.main()
