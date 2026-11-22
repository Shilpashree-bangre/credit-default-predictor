"""Tests run by GitHub Actions before anything is pushed to the Space."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model as m  # noqa: E402


@pytest.fixture(scope="module")
def trained():
    return m.build()


def test_data_has_expected_shape():
    X, y = m.load_data()
    assert list(X.columns) == m.FEATURES
    assert len(X) == 30000
    assert set(y.unique()) == {0, 1}


def test_quality_gate(trained):
    """Block deployment if the model is clearly worse than the Module 8 result (F1 0.532)."""
    _, metrics = trained
    assert metrics["f1"] >= 0.50
    assert metrics["recall"] >= 0.50
    assert metrics["roc_auc"] >= 0.75


def test_prediction_is_a_probability(trained):
    model, _ = trained
    X, _ = m.load_data()
    p = m.predict_one(model, X.iloc[0].to_dict())
    assert 0.0 <= p <= 1.0


def test_missing_input_is_rejected(trained):
    model, _ = trained
    with pytest.raises(ValueError):
        m.predict_one(model, {"LIMIT_BAL": 50000})
