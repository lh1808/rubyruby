"""Tests für Uplift-Metriken (BT und MT)."""

from __future__ import annotations

import numpy as np
import pytest

from rubin.utils.uplift_metrics import (
    uplift_curve,
    auuc,
    qini_coefficient,
    uplift_at_k,
    policy_value,
    _check_binary,
    _check_discrete,
    uplift_curve_mt_per_arm,
    policy_value_mt,
    policy_value_per_arm,
    optimal_treatment_assignment,
    mt_eval_summary,
)


@pytest.fixture
def perfect_data():
    """Perfekter Score: behandelte Responder haben höchsten Score."""
    np.random.seed(42)
    n = 1000
    t = np.array([1] * 500 + [0] * 500)
    y = np.zeros(n, dtype=int)
    y[:200] = 1
    score = np.zeros(n, dtype=float)
    score[:200] = 1.0
    score[200:500] = 0.5
    score[500:] = 0.0
    return y, t, score


@pytest.fixture
def random_data():
    """Zufälliger Score (kein Signal)."""
    np.random.seed(42)
    n = 1000
    t = np.random.binomial(1, 0.5, n)
    y = np.random.binomial(1, 0.3, n)
    score = np.random.uniform(0, 1, n)
    return y, t, score


@pytest.fixture
def mt_data():
    """Multi-Treatment-Daten mit 3 Armen (Control + 2 Treatments)."""
    np.random.seed(42)
    n = 600
    t = np.array([0] * 200 + [1] * 200 + [2] * 200)
    y = np.zeros(n, dtype=int)
    y[:50] = 1       # control responders
    y[200:300] = 1   # treatment 1 responders
    y[400:420] = 1   # treatment 2 responders
    scores_2d = np.random.randn(n, 2)
    scores_2d[:, 0] += 0.5  # T1 besser
    return y, t, scores_2d


class TestUpliftCurve:
    def test_shape(self, random_data):
        y, t, score = random_data
        curve = uplift_curve(y, t, score)
        assert len(curve.fraction) == len(y)
        assert curve.fraction[-1] == pytest.approx(1.0)

    def test_binary_check(self):
        with pytest.raises(ValueError, match="binär"):
            uplift_curve(np.array([0, 1, 2]), np.array([0, 1, 0]), np.array([0.1, 0.2, 0.3]))


class TestAuuc:
    def test_positive_for_good_model(self, perfect_data):
        y, t, score = perfect_data
        curve = uplift_curve(y, t, score)
        assert auuc(curve) > 0

    def test_near_zero_for_random(self, random_data):
        y, t, score = random_data
        curve = uplift_curve(y, t, score)
        assert abs(auuc(curve)) < 0.1


class TestQiniCoefficient:
    def test_positive_for_good_model(self, perfect_data):
        y, t, score = perfect_data
        curve = uplift_curve(y, t, score)
        assert qini_coefficient(curve) > 0


class TestUpliftAtK:
    def test_uplift_at_10pct(self, perfect_data):
        y, t, score = perfect_data
        curve = uplift_curve(y, t, score)
        val = uplift_at_k(curve, k_fraction=0.10)
        assert val > 0

    def test_zero_fraction(self, perfect_data):
        y, t, score = perfect_data
        curve = uplift_curve(y, t, score)
        assert uplift_at_k(curve, k_fraction=0.0) == 0.0


class TestPolicyValue:
    def test_binary_check(self):
        with pytest.raises(ValueError, match="binär"):
            policy_value(np.array([0, 2]), np.array([0, 1]), np.array([0.1, 0.9]))

    def test_all_below_threshold(self, random_data):
        y, t, score = random_data
        val = policy_value(y, t, score, threshold=999.0)
        assert val == 0.0


# --- Multi-Treatment Tests ---

class TestCheckDiscrete:
    def test_valid_discrete(self):
        _check_discrete(np.array([0, 1, 2, 3]), "test")

    def test_non_integer_raises(self):
        with pytest.raises(ValueError, match="ganzzahlig"):
            _check_discrete(np.array([0, 1.5]), "test")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="negative"):
            _check_discrete(np.array([-1, 0, 1]), "test")


class TestUpliftCurveMtPerArm:
    def test_shape(self, mt_data):
        y, t, scores_2d = mt_data
        curve = uplift_curve_mt_per_arm(y, t, scores_2d[:, 0], treatment_arm=1)
        # Nur Control + Arm 1
        n_subset = ((t == 0) | (t == 1)).sum()
        assert len(curve.fraction) == n_subset

    def test_two_arms(self, mt_data):
        y, t, scores_2d = mt_data
        curve1 = uplift_curve_mt_per_arm(y, t, scores_2d[:, 0], treatment_arm=1)
        curve2 = uplift_curve_mt_per_arm(y, t, scores_2d[:, 1], treatment_arm=2)
        # Beide sollten nicht-leere Kurven sein
        assert len(curve1.fraction) > 0
        assert len(curve2.fraction) > 0


class TestPolicyValueMt:
    def test_returns_float(self, mt_data):
        y, t, scores_2d = mt_data
        pv = policy_value_mt(y, t, scores_2d)
        assert isinstance(pv, float)

    def test_with_uniform_propensity(self, mt_data):
        y, t, scores_2d = mt_data
        n = len(y)
        prop = np.ones((n, 3)) / 3.0
        pv = policy_value_mt(y, t, scores_2d, propensity=prop)
        assert isinstance(pv, float)


class TestPolicyValuePerArm:
    def test_returns_float(self, mt_data):
        y, t, scores_2d = mt_data
        pv = policy_value_per_arm(y, t, scores_2d[:, 0], treatment_arm=1)
        assert isinstance(pv, float)

    def test_all_below_threshold(self, mt_data):
        y, t, scores_2d = mt_data
        pv = policy_value_per_arm(y, t, scores_2d[:, 0], treatment_arm=1, threshold=999.0)
        assert pv == 0.0

    def test_consistent_with_bt_policy_value(self):
        """Bei nur einem Treatment-Arm sollte policy_value_per_arm konsistent
        mit der BT-Funktion policy_value sein (gleiche Daten, gleicher Filter)."""
        np.random.seed(42)
        n = 500
        t = np.array([0] * 250 + [1] * 250)
        y = np.random.binomial(1, 0.3, n)
        score = np.random.uniform(-1, 1, n)

        pv_bt = policy_value(y, t, score, threshold=0.0)
        pv_mt = policy_value_per_arm(y, t, score, treatment_arm=1, threshold=0.0)
        assert pv_bt == pytest.approx(pv_mt, abs=1e-10)

    def test_arm2_ignores_arm1(self, mt_data):
        """policy_value_per_arm für Arm 2 sollte nur T=0 und T=2 nutzen."""
        y, t, scores_2d = mt_data
        pv = policy_value_per_arm(y, t, scores_2d[:, 1], treatment_arm=2)
        # Manuell: Nur T in {0, 2}
        mask = (t == 0) | (t == 2)
        y_sub = y[mask]
        t_sub = (t[mask] == 2).astype(int)
        s_sub = scores_2d[:, 1][mask]
        above = s_sub >= 0.0
        y_m = y_sub[above]
        t_m = t_sub[above]
        expected = float(y_m[t_m == 1].mean() - y_m[t_m == 0].mean())
        assert pv == pytest.approx(expected, abs=1e-10)


class TestOptimalTreatmentAssignment:
    def test_shape(self, mt_data):
        _, _, scores_2d = mt_data
        opt = optimal_treatment_assignment(scores_2d)
        assert opt.shape == (len(scores_2d),)

    def test_values_in_range(self, mt_data):
        _, _, scores_2d = mt_data
        opt = optimal_treatment_assignment(scores_2d)
        assert np.all(opt >= 0)
        assert np.all(opt <= 2)

    def test_negative_effects_yield_control(self):
        scores = np.array([[-1.0, -2.0], [-0.5, -0.3]])
        opt = optimal_treatment_assignment(scores)
        np.testing.assert_array_equal(opt, [0, 0])


class TestMtEvalSummary:
    def test_keys(self, mt_data):
        y, t, scores_2d = mt_data
        summary = mt_eval_summary(y, t, scores_2d)
        # Per-Arm-Metriken (Konvention: Kurzform wie in BT)
        assert "qini_T1" in summary
        assert "qini_T2" in summary
        assert "auuc_T1" in summary
        assert "auuc_T2" in summary
        assert "uplift10_T1" in summary
        assert "uplift10_T2" in summary
        assert "uplift20_T1" in summary
        assert "uplift20_T2" in summary
        assert "uplift50_T1" in summary
        assert "uplift50_T2" in summary
        # Per-Arm Policy Values
        assert "policy_value_treat_positive_T1" in summary
        assert "policy_value_treat_positive_T2" in summary
        assert isinstance(summary["policy_value_treat_positive_T1"], float)
        assert isinstance(summary["policy_value_treat_positive_T2"], float)
        # Globale Metriken
        assert "policy_value" in summary
        assert "best_treatment_distribution" in summary
        # Keine BT-only Keys
        assert "qini" not in summary
        assert "uplift_at_10pct" not in summary

    def test_with_propensity(self, mt_data):
        y, t, scores_2d = mt_data
        n = len(y)
        prop = np.ones((n, 3)) / 3.0
        summary = mt_eval_summary(y, t, scores_2d, propensity=prop)
        assert "policy_value" in summary
        assert isinstance(summary["policy_value"], float)

    def test_distribution_sums_to_one(self, mt_data):
        y, t, scores_2d = mt_data
        summary = mt_eval_summary(y, t, scores_2d)
        dist = summary["best_treatment_distribution"]
        total = sum(dist.values())
        assert total == pytest.approx(1.0, abs=1e-6)
