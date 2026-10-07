"""Checks for fold isolation and ordinal scoring in the training workflow."""

import ast
import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from company_reviews import training
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
        search = build_search(
            n_splits=2,
            c_values=[2],
            min_df_values=[1],
            numeric_weight_values=[0.05, 0.1],
            verbose=0,
        )
        search.estimator.set_params(features__char_min_df=1)
        search.fit(reviews, labels, groups=np.arange(len(reviews)))
        model = export_model(search.best_estimator_)
        np.testing.assert_allclose(
            model.predict_proba(reviews), search.predict_proba(reviews)
        )
        self.assertGreaterEqual(model.predict(reviews).min(), 1)
        self.assertLessEqual(model.predict(reviews).max(), 5)
        self.assertEqual(len(search.cv_results_["params"]), 2)
        self.assertEqual(
            model.numeric_weight, search.best_params_["features__numeric_weight"]
        )


def test_search_matches_final_notebook_grid():
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks/baseline.ipynb"
    notebook = json.loads(notebook_path.read_text())
    grid_calls = [
        node
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for node in ast.walk(ast.parse("".join(cell["source"])))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "GridSearchCV"
    ]
    assert len(grid_calls) == 1
    notebook_grid = next(
        ast.literal_eval(keyword.value)
        for keyword in grid_calls[0].keywords
        if keyword.arg == "param_grid"
    )
    search = build_search()
    assert search.param_grid == notebook_grid
    assert len(ParameterGrid(search.param_grid)) == 27
    assert search.cv.n_splits == 5
    assert search.cv.shuffle
    assert search.cv.random_state == 2026
    assert search.refit
    assert search.scoring is negative_median_mae


def test_cli_without_tuning_exports_final_parameters(tmp_path, monkeypatch):
    data = pd.DataFrame(
        [
            {
                "Id": example * 5 + rating,
                "Review": f"{word} service review number {example}",
                "Rating": rating,
            }
            for example in range(8)
            for rating, word in enumerate(
                ["awful", "poor", "okay", "good", "excellent"], start=1
            )
        ]
    )
    data_path = tmp_path / "train.csv"
    model_dir = tmp_path / "models"
    data.to_csv(data_path, index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["training", "--data", str(data_path), "--output", str(model_dir)],
    )

    training.main()

    full = joblib.load(model_dir / "tfidf.joblib")
    assert full.classifier.C == 4
    assert full.word_vectorizer.min_df == 2
    assert full.numeric_weight == 0.05
    assert full.char_vectorizer.min_df == 5
    assert full.char_weight == 0.5
    assert full.numeric_columns == tuple(range(13))
    word = joblib.load(model_dir / "word-only.joblib")
    assert word.classifier.C == 4
    assert word.word_vectorizer.min_df == 5
    assert word.numeric_weight == 0
    assert word.char_vectorizer is None


def test_pipeline_defaults_preserve_notebook_feature_experiments():
    pipeline = build_pipeline()
    features = pipeline.named_steps["features"]
    assert features.min_df == 5
    assert features.numeric_weight == 0.1
    assert features.char_weight == 0.5
    assert pipeline.named_steps["classifier"].C == 4


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
