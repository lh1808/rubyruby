"""Tests für Feature-Selektion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rubin.feature_selection import (
    remove_highly_correlated_features,
    select_features_by_importance,
    _top_pct_features,
)


class TestRemoveHighlyCorrelatedFeatures:
    """Korrelationsfilter mit importance-gesteuerter Auswahl."""

    def _make_correlated_df(self, n=200):
        rng = np.random.RandomState(42)
        base = rng.randn(n)
        return pd.DataFrame({
            "a": base,
            "b": base + rng.randn(n) * 0.01,  # ~1.0 corr with a
            "c": rng.randn(n),                  # uncorrelated
            "d": rng.randn(n),                  # uncorrelated
        })

    def test_removes_correlated_pair(self):
        X = self._make_correlated_df()
        importances = {"m": pd.Series({"a": 10, "b": 5, "c": 3, "d": 1})}
        result, removed, absorbed = remove_highly_correlated_features(
            X, threshold=0.9, importances=importances
        )
        assert "a" in result.columns  # higher importance survives
        assert "b" not in result.columns
        assert "b" in removed

    def test_absorbed_by_mapping(self):
        X = self._make_correlated_df()
        importances = {"m": pd.Series({"a": 10, "b": 5, "c": 3, "d": 1})}
        _, _, absorbed = remove_highly_correlated_features(
            X, threshold=0.9, importances=importances
        )
        assert absorbed.get("b") == "a"

    def test_keeps_uncorrelated(self):
        X = self._make_correlated_df()
        importances = {"m": pd.Series({"a": 10, "b": 5, "c": 3, "d": 1})}
        result, removed, _ = remove_highly_correlated_features(
            X, threshold=0.9, importances=importances
        )
        assert "c" in result.columns
        assert "d" in result.columns

    def test_high_threshold_removes_nothing(self):
        X = self._make_correlated_df()
        importances = {"m": pd.Series({"a": 10, "b": 5, "c": 3, "d": 1})}
        result, removed, absorbed = remove_highly_correlated_features(
            X, threshold=0.999, importances=importances
        )
        assert len(removed) == 0
        assert len(absorbed) == 0

    def test_single_column(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        result, removed, absorbed = remove_highly_correlated_features(X, threshold=0.9)
        assert list(result.columns) == ["a"]
        assert removed == []

    def test_without_importances(self):
        """Without importances, one of correlated pair is still removed."""
        X = self._make_correlated_df()
        result, removed, absorbed = remove_highly_correlated_features(
            X, threshold=0.9
        )
        # One of a/b should be removed
        assert len(removed) == 1
        assert removed[0] in ("a", "b")

    def test_categorical_columns_skipped(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({
            "num1": rng.randn(100),
            "num2": rng.randn(100),
            "cat1": pd.Categorical(rng.choice(["x", "y"], 100)),
        })
        result, removed, _ = remove_highly_correlated_features(X, threshold=0.5)
        assert "cat1" in result.columns  # categorical always kept


class TestTopPctFeatures:
    def test_top_10pct_of_10(self):
        imp = pd.Series({"a": 10, "b": 9, "c": 8, "d": 7, "e": 6, "f": 5, "g": 4, "h": 3, "i": 2, "j": 1})
        result = _top_pct_features(imp, top_pct=10.0, n_total=10)
        assert result == ["a"]

    def test_top_50pct_of_4(self):
        imp = pd.Series({"a": 10, "b": 5, "c": 3, "d": 1})
        result = _top_pct_features(imp, top_pct=50.0, n_total=4)
        assert len(result) == 2
        assert "a" in result and "b" in result

    def test_top_pct_at_least_one(self):
        imp = pd.Series({"a": 10, "b": 5, "c": 3})
        result = _top_pct_features(imp, top_pct=1.0, n_total=3)
        assert len(result) >= 1


class TestSelectFeaturesByImportance:
    def test_single_method(self):
        X = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        importances = {"lgbm": pd.Series({"a": 10, "b": 5, "c": 3, "d": 1})}
        result, removed, top = select_features_by_importance(X, importances, top_pct=50.0)
        assert "a" in result.columns
        assert "b" in result.columns
        assert len(removed) == 2

    def test_union_of_two_methods(self):
        X = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        importances = {
            "method1": pd.Series({"a": 10, "b": 1, "c": 1, "d": 1}),
            "method2": pd.Series({"a": 1, "b": 1, "c": 1, "d": 10}),
        }
        # Top 25% = 1 feature each: method1→a, method2→d
        result, removed, top = select_features_by_importance(X, importances, top_pct=25.0)
        assert "a" in result.columns
        assert "d" in result.columns
        assert len(result.columns) == 2

    def test_max_features_caps_union(self):
        X = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        importances = {
            "m1": pd.Series({"a": 10, "b": 8, "c": 1, "d": 1}),
            "m2": pd.Series({"a": 1, "b": 1, "c": 8, "d": 10}),
        }
        # Top 50% = 2 per method: m1→a,b; m2→c,d → union=4
        # max_features=2 should cap it
        result, removed, top = select_features_by_importance(
            X, importances, top_pct=50.0, max_features=2
        )
        assert len(result.columns) == 2

    def test_empty_importances(self):
        X = pd.DataFrame({"a": [1], "b": [2]})
        result, removed, top = select_features_by_importance(X, {}, top_pct=50.0)
        assert len(result.columns) == 2
        assert removed == []

    def test_preserves_column_order(self):
        X = pd.DataFrame({"z": [1], "a": [2], "m": [3]})
        importances = {"m1": pd.Series({"z": 10, "a": 5, "m": 1})}
        # ceil(50% × 3) = ceil(1.5) = 2 → behält z, a (höchste Importance)
        result, _, _ = select_features_by_importance(X, importances, top_pct=50.0)
        assert list(result.columns) == ["z", "a"]
