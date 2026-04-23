"""
hed/metrics.py
--------------
Evaluation metrics for time-series early detection.

Provides:
  auc_score       : standard ROC-AUC (temporally agnostic baseline)
  far_at_threshold: false alarm rate at a given decision threshold
  hed_far_curve   : HED vs FAR curve (the paper's key evaluation plot)

These are kept deliberately simple so HED can be compared against
AUC and FAR without external dependencies beyond NumPy/scikit-learn.
"""

import numpy as np
from typing import Tuple

# Optional scikit-learn import — graceful fallback if not installed.
try:
    from sklearn.metrics import roc_auc_score as _sklearn_auc
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------

def auc_score(
    P: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute the Area Under the ROC Curve (AUC).

    AUC is a temporally agnostic metric: it treats a detection at
    t = t* + 1 the same as a detection at t = t* + 100.  This is the
    key inadequacy that HED is designed to resolve.

    Parameters
    ----------
    P : np.ndarray, shape (T,)
        Predicted probability stream (posterior of the anomalous regime).
    labels : np.ndarray, shape (T,)
        Binary ground-truth labels (1 = anomalous, 0 = normal).

    Returns
    -------
    float
        AUC value in [0, 1].

    Notes
    -----
    Uses sklearn.metrics.roc_auc_score if available, otherwise falls
    back to a pure-NumPy trapezoidal implementation.
    """
    P = np.asarray(P, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    if len(np.unique(labels)) < 2:
        # All labels the same — AUC is undefined; return 0.5 (chance level).
        return 0.5

    if _SKLEARN_AVAILABLE:
        return float(_sklearn_auc(labels, P))
    else:
        return _numpy_auc(P, labels)


def _numpy_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Pure NumPy ROC-AUC via trapezoidal rule."""
    # Sort by descending score
    desc_idx = np.argsort(-scores)
    labels_sorted = labels[desc_idx]

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr = np.concatenate([[0.0], np.cumsum(labels_sorted == 1) / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], np.cumsum(labels_sorted == 0) / n_neg, [1.0]])

    return float(np.trapz(tpr, fpr))


# ---------------------------------------------------------------------------
# False Alarm Rate
# ---------------------------------------------------------------------------

def far_at_threshold(
    P: np.ndarray,
    t_star: int,
    threshold: float,
) -> float:
    """Compute the False Alarm Rate (FAR) at a decision threshold.

    FAR = (# time steps before t_star where P >= threshold) / t_star

    Parameters
    ----------
    P : np.ndarray, shape (T,)
        Predicted probability stream.
    t_star : int
        Index of the true regime-shift onset.
    threshold : float
        Decision threshold in [0, 1].

    Returns
    -------
    float
        FAR in [0, 1].
    """
    P = np.asarray(P, dtype=np.float64)
    if t_star == 0:
        return 0.0
    pre_shift = P[:t_star]
    return float(np.mean(pre_shift >= threshold))


# ---------------------------------------------------------------------------
# HED vs FAR curve
# ---------------------------------------------------------------------------

def hed_far_curve(
    P: np.ndarray,
    t_star: int,
    lam: float = 0.1,
    n_thresholds: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the HED–FAR curve over a range of decision thresholds.

    At each threshold τ:
    - The probability stream is *thresholded*: P_τ(t) = 1 if P(t) >= τ, else 0.
    - HED(P_τ) is computed.
    - FAR(P_τ) is computed as the fraction of pre-shift steps above τ.

    This produces a trade-off curve analogous to the ROC curve but
    measuring *temporal* quality (HED) against false alarm rate (FAR).

    A good detector has high HED and low FAR — the curve should be
    as far to the upper-left as possible.

    Parameters
    ----------
    P : np.ndarray, shape (T,)
        Predicted probability stream.
    t_star : int
        Index of the true regime-shift onset.
    lam : float
        Hiremath Decay Constant.
    n_thresholds : int
        Number of threshold values to sweep.

    Returns
    -------
    far_values : np.ndarray, shape (n_thresholds,)
    hed_values : np.ndarray, shape (n_thresholds,)

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> from hed.metrics import hed_far_curve
    >>> far, hed = hed_far_curve(P, t_star=100, lam=0.1)
    >>> plt.plot(far, hed)
    >>> plt.xlabel("FAR"); plt.ylabel("HED"); plt.show()
    """
    from .core import hed_score_discrete   # local import to avoid circularity

    P = np.asarray(P, dtype=np.float64)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    far_values = np.empty(n_thresholds)
    hed_values = np.empty(n_thresholds)

    for i, tau in enumerate(thresholds):
        P_bin = (P >= tau).astype(np.float64)
        far_values[i] = far_at_threshold(P, t_star, tau)
        # Use non-normalised HED on binary stream for cleaner FAR comparison
        hed_values[i] = hed_score_discrete(P_bin, t_star, lam=lam, normalise=True)

    return far_values, hed_values


# ---------------------------------------------------------------------------
# Convenience: head-to-head comparison table
# ---------------------------------------------------------------------------

def compare_detectors(
    detectors: dict,
    t_star: int,
    T: int,
    lam: float = 0.1,
) -> dict:
    """Compare multiple detectors on HED and AUC simultaneously.

    Parameters
    ----------
    detectors : dict of {name: P_array}
        Mapping from detector name to its probability stream.
    t_star : int
        Regime-shift onset.
    T : int
        Total number of time steps.
    lam : float
        Hiremath Decay Constant.

    Returns
    -------
    dict of {name: {'HED': float, 'AUC': float}}
    """
    from .core import hed_score_discrete

    labels = np.zeros(T, dtype=np.int32)
    labels[t_star:] = 1

    results = {}
    for name, P in detectors.items():
        P = np.asarray(P, dtype=np.float64)
        results[name] = {
            "HED": hed_score_discrete(P, t_star, lam=lam, normalise=True),
            "AUC": auc_score(P, labels),
        }
    return results
