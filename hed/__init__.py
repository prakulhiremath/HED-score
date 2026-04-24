"""
hed: Hiremath Early Detection Score
====================================

A principled, measure-theoretic metric for evaluating early detection
performance in time-series anomaly/regime-change tasks.

Unlike AUC, HED is temporally aware: it rewards detectors that raise
probability mass *early* after a regime shift, and penalises those
that detect late—even when their overall discrimination is identical.

Reference
---------
Hiremath, P. S. (2026). The Hiremath Early Detection (HED) Score:
A Measure-Theoretic Evaluation Standard for Temporal Intelligence.
arXiv:2604.04993 [stat.ML].

Quick start
-----------
>>> import numpy as np
>>> from hed import hed_score
>>> T = 200
>>> t_star = 100                          # regime shift at t=100
>>> P = np.zeros(T)
>>> P[t_star:] = np.linspace(0, 1, T - t_star)
>>> score = hed_score(P, t_star, lam=0.1)
>>> print(f"HED = {score:.4f}")
"""

from .core import hed_score, hed_score_discrete, hed_score_continuous
from .metrics import auc_score, far_at_threshold, hed_far_curve
from .utils import baseline_correct, exponential_kernel, smooth_probabilities

__version__ = "0.1.1"
__author__ = "Prakul Sunil Hiremath"
__license__ = "Apache 2.0"

__all__ = [
    # Core HED
    "hed_score",
    "hed_score_discrete",
    "hed_score_continuous",
    # Metrics
    "auc_score",
    "far_at_threshold",
    "hed_far_curve",
    # Utilities
    "baseline_correct",
    "exponential_kernel",
    "smooth_probabilities",
]
