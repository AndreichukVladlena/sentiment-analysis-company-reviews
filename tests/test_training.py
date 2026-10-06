"""Checks for fold isolation and ordinal scoring in the training workflow."""

import json
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone

from company_reviews.training import (
    ReviewFeatures,
    build_pipeline,
    build_search,
    export_model,
    negative_median_mae,
    save_models,
)


class TrainingTest(unittest.TestCase):
    def test_validation_does_not_change_vocabulary_or_scaler(self):
        train = pd.Series(["good service!", "bad service", "fine service"])
        features = ReviewFeatures(min_df=1, char_min_df=1).fit(train)
        before = features.scaler_.mean_.copy()
        vocabulary = features.word_vectorizer_.vocabulary_.copy()
        features.transform(["unseenvalidationword " * 100])
        self.assertEqual(features.word_vectorizer_.vocabulary_, vocabulary)
        self.assertNotIn("unseenvalidationword", vocabulary)
        np.testing.assert_array_equal(features.scaler_.mean_, before)
        self.assertEqual(clone(features).min_df, 1)

    def test_scorer_uses_probability_median_and_real_class_labels(self):
        class Distribution:
            classes_ = np.array([1, 3, 5])

            def predict_proba(self, reviews):
                return np.array([[0.4, 0.2, 0.4], [0.1, 0.2, 0.7]])

        self.assertEqual(negative_median_mae(Distribution(), ["a", "b"], [3, 4]), -0.5)

    def test_grid_search_and_export_preserve_predictions(self):
        reviews = pd.Series(
            [
                f"{word} service review number {i}"
                for i in range(5)
                for word in ["awful", "okay", "great"]
            ]
        )
        labels = np.tile([1, 3, 5], 5)
        search = build_search(n_splits=2, c_values=[2], min_df_values=[1], verbose=0)
        search.estimator.set_params(features__char_min_df=1)
        search.fit(reviews, labels, groups=np.arange(len(reviews)))
        model = export_model(search.best_estimator_)
        np.testing.assert_allclose(
            model.predict_proba(reviews), search.predict_proba(reviews)
        )
        self.assertGreaterEqual(model.predict(reviews).min(), 1)
        self.assertLessEqual(model.predict(reviews).max(), 5)


def test_baseline_reexport_preserves_deberta_and_selected_default(tmp_path):
    model = build_pipeline(min_df=1, word_only=True)
    model.fit(
        ["bad delivery", "good delivery", "bad service", "good service"], [1, 5, 1, 5]
    )
    save_models(model, model, tmp_path)
    path = tmp_path / "manifest.json"
    catalog = json.loads(path.read_text())
    assert catalog["default_model"] == "tfidf"
    extra = {"id": "deberta", "version": "full-existing", "filename": "deberta"}
    catalog["models"].append(extra)
    catalog["default_model"] = "deberta"
    path.write_text(json.dumps(catalog))
    save_models(model, model, tmp_path)
    result = json.loads(path.read_text())
    assert result["default_model"] == "deberta"
    assert [entry["id"] for entry in result["models"]] == [
        "tfidf",
        "word-only",
        "deberta",
    ]
    assert result["models"][2] == extra


if __name__ == "__main__":
    unittest.main()
