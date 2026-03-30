"""Tests für die Model Registry."""

from __future__ import annotations

import numpy as np
import pytest

from rubin.model_registry import ModelRegistry, ModelContext, default_registry
from rubin.model_management import ModelEntry, choose_champion
from rubin.training import _predict_effect


class TestModelRegistry:
    def test_default_registry_contains_all_models(self):
        reg = default_registry()
        names = reg.list()
        expected = ["CausalForestDML", "DRLearner", "NonParamDML", "ParamDML",
                    "SLearner", "TLearner", "XLearner"]
        assert names == expected

    def test_unknown_model_raises(self):
        reg = default_registry()
        with pytest.raises(KeyError, match="Unbekanntes Modell"):
            reg.create("DoesNotExist", ModelContext())

    def test_create_slearner(self):
        reg = default_registry()
        model = reg.create("SLearner", ModelContext(base_learner_type="lgbm"))
        assert hasattr(model, "fit")
        assert hasattr(model, "effect")

    def test_create_nonparamdml_has_discrete_flags(self):
        """Prüft, dass NonParamDML mit discrete_treatment=True erstellt wird."""
        reg = default_registry()
        model = reg.create("NonParamDML", ModelContext(base_learner_type="lgbm"))
        assert getattr(model, "discrete_treatment", False) is True
        assert getattr(model, "discrete_outcome", False) is True

    def test_create_drlearner_has_cv_and_random_state(self):
        """Prüft, dass DRLearner mit cv und random_state erstellt wird."""
        reg = default_registry()
        ctx = ModelContext(base_learner_type="lgbm", seed=123)
        model = reg.create("DRLearner", ctx)
        assert getattr(model, "random_state", None) == 123
        assert getattr(model, "cv", None) == 5

    def test_create_causalforestdml_has_discrete_flags(self):
        reg = default_registry()
        model = reg.create("CausalForestDML", ModelContext(base_learner_type="lgbm"))
        assert getattr(model, "discrete_treatment", False) is True
        assert getattr(model, "discrete_outcome", False) is True


class TestChooseChampion:
    def test_highest_metric(self):
        entries = [
            ModelEntry(name="A", artifact_path="a.pkl", metrics={"qini": 0.5}),
            ModelEntry(name="B", artifact_path="b.pkl", metrics={"qini": 0.8}),
            ModelEntry(name="C", artifact_path="c.pkl", metrics={"qini": 0.3}),
        ]
        assert choose_champion(entries, metric="qini", higher_is_better=True) == "B"

    def test_lowest_metric(self):
        entries = [
            ModelEntry(name="A", artifact_path="a.pkl", metrics={"loss": 0.5}),
            ModelEntry(name="B", artifact_path="b.pkl", metrics={"loss": 0.1}),
        ]
        assert choose_champion(entries, metric="loss", higher_is_better=False) == "B"

    def test_missing_metric_ignored(self):
        entries = [
            ModelEntry(name="A", artifact_path="a.pkl", metrics={}),
            ModelEntry(name="B", artifact_path="b.pkl", metrics={"qini": 0.5}),
        ]
        assert choose_champion(entries, metric="qini") == "B"

    def test_empty_list_returns_none(self):
        assert choose_champion([], metric="qini") is None


class TestPredictEffect:
    def test_prefers_const_marginal_effect(self):
        """Wenn ein Modell beide Methoden hat, soll const_marginal_effect bevorzugt werden."""
        import pandas as pd

        class DualModel:
            def const_marginal_effect(self, X):
                return np.ones(len(X)) * 0.5

            def effect(self, X):
                return np.ones(len(X)) * 0.9

        model = DualModel()
        X = pd.DataFrame({"a": [1, 2, 3]})
        result = _predict_effect(model, X)
        assert np.allclose(result, 0.5)

    def test_falls_back_to_effect(self):
        """Wenn nur effect() existiert, soll diese verwendet werden."""
        import pandas as pd

        class EffectOnlyModel:
            def effect(self, X):
                return np.ones(len(X)) * 0.7

        model = EffectOnlyModel()
        X = pd.DataFrame({"a": [1, 2]})
        result = _predict_effect(model, X)
        assert np.allclose(result, 0.7)

    def test_raises_without_either(self):
        import pandas as pd

        class NoMethodModel:
            pass

        with pytest.raises(AttributeError):
            _predict_effect(NoMethodModel(), pd.DataFrame({"a": [1]}))
