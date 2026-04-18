"""Tests for src/models/explainer.py.

Uses a real tiny XGBoost (5 estimators, max_depth=2, 20 rows) so SHAP
TreeExplainer has a genuine tree structure to walk. No DuckDB or MLflow
required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from src.models.explainer import explain

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_FEATURE_NAMES = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]


@pytest.fixture(scope="module")
def tiny_model_and_X():
    """Fit a tiny XGBoost on 20 rows and return (model, X_test).

    Module-scoped so the model is trained once and shared across tests.
    """
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.random((20, 5)), columns=_FEATURE_NAMES)
    y_train = rng.random(20) * 100
    model = XGBRegressor(n_estimators=5, max_depth=2, random_state=0)
    model.fit(X_train, y_train)

    X_test = pd.DataFrame(rng.random((6, 5)), columns=_FEATURE_NAMES)
    return model, X_test


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_explain_returns_one_dict_per_row(tiny_model_and_X) -> None:
    """explain() returns exactly one dict per row in X."""
    model, X = tiny_model_and_X
    result = explain(model, X)
    assert len(result) == len(X)


def test_explain_dict_keys_match_feature_names(tiny_model_and_X) -> None:
    """Keys in every returned dict match the column names of X."""
    model, X = tiny_model_and_X
    result = explain(model, X)
    expected_keys = set(_FEATURE_NAMES)
    for i, row_dict in enumerate(result):
        assert set(row_dict.keys()) == expected_keys, (
            f"Row {i}: expected keys {expected_keys}, got {set(row_dict.keys())}"
        )


def test_explain_values_are_floats(tiny_model_and_X) -> None:
    """All SHAP values are Python floats (not numpy scalars)."""
    model, X = tiny_model_and_X
    result = explain(model, X)
    for row_dict in result:
        for feat, val in row_dict.items():
            assert isinstance(val, float), (
                f"Expected float for feature {feat!r}, got {type(val).__name__}"
            )


def test_explain_values_rounded_to_4dp(tiny_model_and_X) -> None:
    """All SHAP values are rounded to exactly 4 decimal places."""
    model, X = tiny_model_and_X
    result = explain(model, X)
    for row_dict in result:
        for feat, val in row_dict.items():
            assert val == round(val, 4), (
                f"Feature {feat!r} value {val} is not rounded to 4 dp"
            )


def test_explain_single_row(tiny_model_and_X) -> None:
    """explain() handles a single-row DataFrame without error."""
    model, X = tiny_model_and_X
    X_single = X.iloc[[0]]
    result = explain(model, X_single)
    assert len(result) == 1
    assert set(result[0].keys()) == set(_FEATURE_NAMES)


def test_explain_raises_on_empty_dataframe(tiny_model_and_X) -> None:
    """explain() raises ValueError on an empty DataFrame."""
    model, _ = tiny_model_and_X
    X_empty = pd.DataFrame(columns=_FEATURE_NAMES)
    with pytest.raises(ValueError, match="empty"):
        explain(model, X_empty)
