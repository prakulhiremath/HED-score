"""
experiments/nsl_kdd.py
-----------------------
Replicates the NSL-KDD cyber-intrusion detection experiment from the paper.

NSL-KDD is a standard benchmark for network intrusion detection.
We treat it as a regime-shift problem:
  - Normal connections → regime 0
  - Attack connections  → regime 1

Since NSL-KDD is not bundled in this repo, this script:
  1. Attempts to load from data/nsl_kdd_test.csv (place it there yourself).
  2. If not found, generates a synthetic stand-in that mirrors the paper's
     described statistics, so the experiment can always be run.

Download NSL-KDD from: https://www.unb.ca/cic/datasets/nsl.html

Usage:
    python experiments/nsl_kdd.py [--synthetic]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hed import hed_score
from hed.metrics import auc_score, compare_detectors
from hed.utils import sigmoid


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_nsl_kdd(path: str = "data/nsl_kdd_test.csv"):
    """Load NSL-KDD data and return (features, labels).

    Expected CSV columns:
        duration, protocol_type, ..., class
    where class ∈ {'normal', 'neptune', 'smurf', ...}

    Returns None if file not found.
    """
    if not os.path.exists(path):
        return None

    try:
        import pandas as pd
    except ImportError:
        print("[WARNING] pandas not installed; cannot load NSL-KDD CSV.")
        return None

    df = pd.read_csv(path, header=None)

    # NSL-KDD column structure (41 features + label + difficulty)
    if df.shape[1] >= 42:
        labels_raw = df.iloc[:, 41].astype(str).str.strip().str.lower()
    else:
        labels_raw = df.iloc[:, -1].astype(str).str.strip().str.lower()

    labels = (labels_raw != "normal").astype(int).values

    # Use a subset of numeric features
    numeric_cols = [0] + list(range(4, 11)) + list(range(22, 31))
    feats = df.iloc[:, [c for c in numeric_cols if c < df.shape[1]]].values.astype(float)

    return feats, labels


# ---------------------------------------------------------------------------
# Simulated detector (Random Forest log-odds proxy)
# ---------------------------------------------------------------------------

def simulate_rf_detector(labels: np.ndarray, noise: float = 0.15, seed: int = 42):
    """Simulate an RF detector's probability stream.

    For reproducibility when no real model is trained.
    Adds calibrated noise to the ground-truth labels.
    """
    rng = np.random.default_rng(seed)
    log_odds = 2.0 * labels.astype(float) - 1.0   # -1 for normal, +1 for attack
    log_odds += rng.normal(0, noise, size=len(labels))
    return sigmoid(log_odds)


def simulate_lstm_detector(labels: np.ndarray, delay: int = 5, seed: int = 7):
    """Simulate an LSTM detector with detection lag."""
    rng = np.random.default_rng(seed)
    # LSTM sees the shift slightly delayed
    shifted = np.roll(labels, delay).astype(float)
    shifted[:delay] = 0
    log_odds = 2.0 * shifted - 1.0
    log_odds += rng.normal(0, 0.2, size=len(labels))
    return sigmoid(log_odds)


# ---------------------------------------------------------------------------
# Synthetic NSL-KDD stand-in
# ---------------------------------------------------------------------------

def make_synthetic_nsl(T: int = 5000, t_star: int = 2000, seed: int = 0):
    """Generate synthetic data that mirrors NSL-KDD structure."""
    rng = np.random.default_rng(seed)
    labels = np.zeros(T, dtype=int)
    labels[t_star:] = 1
    # Simulate a realistic attack pattern: not all steps are attacks
    attack_rate = rng.binomial(1, 0.7, size=T - t_star)
    labels[t_star:] = attack_rate
    return labels, t_star


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_nsl_kdd_experiment(use_synthetic: bool = False, lam: float = 0.3):
    """Run the NSL-KDD early detection experiment.

    Parameters
    ----------
    use_synthetic : bool
        Force use of synthetic data even if real data is present.
    lam : float
        Hiremath Decay Constant.  Use 0.3 for cyber-physical security
        per the Hiremath Standard Table.
    """
    print("\n" + "=" * 60)
    print("  NSL-KDD Intrusion Detection Experiment")
    print(f"  λ = {lam}  (cyber-physical security setting)")
    print("=" * 60)

    # --- Load or simulate data ---
    data = None if use_synthetic else load_nsl_kdd()

    if data is not None:
        feats, labels = data
        T = len(labels)
        # Find first attack to define t_star
        attack_idx = np.where(labels == 1)[0]
        t_star = int(attack_idx[0]) if len(attack_idx) > 0 else T // 2
        print(f"  Loaded real NSL-KDD: T={T}, t*={t_star}")
    else:
        labels, t_star = make_synthetic_nsl()
        T = len(labels)
        print(f"  Using synthetic NSL-KDD stand-in: T={T}, t*={t_star}")
        print("  [Place data/nsl_kdd_test.csv to use real data]")

    # --- Simulate detectors ---
    P_rf = simulate_rf_detector(labels)
    P_lstm = simulate_lstm_detector(labels, delay=10)

    detectors = {
        "RF (simulated)": P_rf,
        "LSTM (simulated)": P_lstm,
    }

    results = compare_detectors(detectors, t_star=t_star, T=T, lam=lam)

    # --- Print results ---
    print(f"\n  {'Detector':<22} {'AUC':>8} {'HED':>10}")
    print("  " + "-" * 44)
    for name, scores in results.items():
        print(f"  {name:<22} {scores['AUC']:>8.4f} {scores['HED']:>10.4f}")

    print()
    print("  Note: In cyber-physical security, every timestep of early")
    print("  detection matters.  HED (λ=0.3) captures this urgency;")
    print("  AUC does not.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Force use of synthetic data")
    parser.add_argument("--lam", type=float, default=0.3,
                        help="Hiremath Decay Constant")
    args = parser.parse_args()
    run_nsl_kdd_experiment(use_synthetic=args.synthetic, lam=args.lam)
