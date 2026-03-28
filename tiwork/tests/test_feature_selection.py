"""Tests für Feature-Selektion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rubin.feature_selection import (
    remove_low_importance_features,
    select_features_by_importance,
    _top_pct_features,
)


class TestRemoveLowImportanceFeatures:
    """Legacy-Schnittstelle."""

    def test_basic_threshold(self):
        X = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        imp = pd.Series({"a": 100.0, "b": 50.0, "c": 1.0})
        result, removed = remove_low_importance_features(X, imp, importance_threshold_pct_of_max=10.0)
        assert "a" in result.columns
        assert "b" in result.columns
        assert "c" not in result.columns
        assert "c" in removed

    def test_max_features_limits_output(self):
        X = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        imp = pd.Series({"a": 100.0, "b": 80.0, "c": 50.0, "d": 30.0})
        result, removed = remove_low_importance_features(
            X, imp, importance_threshold_pct_of_max=0.0, max_features=2
        )
        assert len(result.columns) == 2
        assert "a" in result.columns
        assert "b" in result.columns
        assert len(removed) == 2

    def test_max_features_none_keeps_all(self):
        X = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        imp = pd.Series({"a": 100.0, "b": 80.0, "c": 50.0})
        result, removed = remove_low_importance_features(
            X, imp, importance_threshold_pct_of_max=0.0, max_features=None
        )
        assert len(result.columns) == 3
        assert removed == []

    def test_empty_importance(self):
        X = pd.DataFrame({"a": [1], "b": [2]})
        imp = pd.Series(dtype=float)
        result, removed = remove_low_importance_features(X, imp)
        assert list(result.columns) == ["a", "b"]
        assert removed == []


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
