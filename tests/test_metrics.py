"""
tests/test_metrics.py
---------------------
Unit tests for metrics: AUC, FAR, HED-FAR curve.
"""

import numpy as np
import pytest

from hed.metrics import auc_score, far_at_threshold, hed_far_curve, compare_detectors


class TestAUC:

    def test_perfect_auc_is_one(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        P = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
        assert abs(auc_score(P, labels) - 1.0) < 1e-6

    def test_random_auc_is_half(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        P = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        # All same score → AUC = 0.5
        assert abs(auc_score(P, labels) - 0.5) < 0.01

    def test_single_class_returns_half(self):
        labels = np.zeros(100, dtype=int)
        P = np.random.default_rng(0).uniform(0, 1, 100)
        assert auc_score(P, labels) == 0.5


class TestFAR:

    def test_no_alarms_before_shift(self):
        T, t_star = 200, 100
        P = np.zeros(T)
        P[t_star:] = 0.9
        assert far_at_threshold(P, t_star, threshold=0.5) == 0.0

    def test_all_alarms_before_shift(self):
        T, t_star = 200, 100
        P = np.ones(T) * 0.9
        assert abs(far_at_threshold(P, t_star, threshold=0.5) - 1.0) < 1e-9

    def test_zero_t_star_returns_zero(self):
        P = np.ones(100)
        assert far_at_threshold(P, t_star=0, threshold=0.5) == 0.0


class TestHEDFARCurve:

    def test_output_shapes(self):
        T, t_star = 200, 80
        P = np.zeros(T)
        P[t_star:] = 0.9
        far, hed = hed_far_curve(P, t_star, lam=0.1, n_thresholds=50)
        assert len(far) == 50
        assert len(hed) == 50

    def test_far_in_zero_one(self):
        T, t_star = 200, 80
        P = np.random.default_rng(0).uniform(0, 1, T)
        far, hed = hed_far_curve(P, t_star, lam=0.1, n_thresholds=30)
        assert np.all(far >= 0) and np.all(far <= 1.0)


class TestCompareDetectors:

    def test_returns_all_keys(self):
        T, t_star = 200, 80
        detectors = {
            "A": np.zeros(T),
            "B": np.ones(T) * 0.5,
        }
        results = compare_detectors(detectors, t_star=t_star, T=T, lam=0.1)
        assert set(results.keys()) == {"A", "B"}
        for name in results:
            assert "HED" in results[name]
            assert "AUC" in results[name]
