"""
experiments/financial_ts.py
----------------------------
Placeholder for financial time-series early detection experiment.

Demonstrates HED applied to market regime-change detection
(e.g., detecting the onset of a crash or volatility spike).

This script is structured and runnable with synthetic data.
Replace the data loading section with real OHLCV data
(e.g., from yfinance or a CSV) for production use.

Usage:
    python experiments/financial_ts.py
    python experiments/financial_ts.py --ticker SPY --start 2007-01-01
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hed import hed_score
from hed.metrics import auc_score, compare_detectors
from hed.utils import sigmoid, smooth_probabilities


# ---------------------------------------------------------------------------
# Synthetic market regime data
# ---------------------------------------------------------------------------

def make_market_data(
    T: int = 500,
    t_star: int = 250,
    pre_vol: float = 0.01,
    post_vol: float = 0.04,
    seed: int = 42,
):
    """Generate synthetic log-returns with a volatility regime change.

    Parameters
    ----------
    T : int
        Total number of trading days.
    t_star : int
        Day of regime change (e.g., start of a crash).
    pre_vol : float
        Normal-period daily volatility.
    post_vol : float
        Crisis-period daily volatility.
    seed : int

    Returns
    -------
    returns : np.ndarray, shape (T,)
    labels : np.ndarray, shape (T,), binary (1 = crisis)
    t_star : int
    """
    rng = np.random.default_rng(seed)
    returns = np.empty(T)
    returns[:t_star] = rng.normal(0.0005, pre_vol, size=t_star)
    returns[t_star:] = rng.normal(-0.002, post_vol, size=T - t_star)

    labels = np.zeros(T, dtype=int)
    labels[t_star:] = 1

    return returns, labels, t_star


# ---------------------------------------------------------------------------
# Detectors (returns → posterior probability of crisis)
# ---------------------------------------------------------------------------

def volatility_detector(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling volatility detector.

    Uses z-score of rolling std relative to long-run std as anomaly score.
    High rolling volatility → high posterior probability of crisis.
    """
    T = len(returns)
    P = np.zeros(T)
    long_vol = np.std(returns)

    for t in range(window, T):
        rolling_vol = np.std(returns[t - window:t])
        z = (rolling_vol - long_vol) / (long_vol + 1e-8)
        P[t] = sigmoid(np.array([z]))[0]

    return np.clip(P, 0.0, 1.0)


def cusum_detector(returns: np.ndarray, k: float = 0.5, h: float = 5.0) -> np.ndarray:
    """CUSUM change-point detector mapped to probability.

    Accumulates signed deviations below the long-run mean.
    A large negative CUSUM suggests a downward regime shift.
    """
    T = len(returns)
    mu = np.mean(returns)
    S = np.zeros(T)
    cusum = 0.0
    S_max = max(h, 1.0)  # for normalisation

    for t in range(1, T):
        cusum = max(0.0, cusum - (returns[t] - mu) + k * np.std(returns))
        S[t] = cusum
        S_max = max(S_max, cusum)

    # Map CUSUM → probability via sigmoid
    return sigmoid(S / S_max * 6 - 3)   # scale to roughly [-3, 3]


# ---------------------------------------------------------------------------
# Optional: real data via yfinance
# ---------------------------------------------------------------------------

def load_real_data(ticker: str, start: str, end: str = None):
    """Load real OHLCV data via yfinance (optional dependency).

    Returns (returns, t_star_estimate) or None if yfinance unavailable.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    import pandas as pd
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        return None

    close = data["Close"].values.flatten()
    returns = np.diff(np.log(close))

    # Estimate t_star as the day with the largest single-day drop
    t_star = int(np.argmin(returns))
    labels = np.zeros(len(returns), dtype=int)
    labels[t_star:] = 1

    return returns, labels, t_star


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_financial_experiment(
    ticker: str = None,
    start: str = "2007-01-01",
    lam: float = 0.2,
):
    """Run the financial time-series experiment.

    Parameters
    ----------
    ticker : str or None
        If provided, load real data via yfinance.
    start : str
        Start date for real data download.
    lam : float
        Hiremath Decay Constant.  0.2 recommended for financial surveillance.
    """
    print("\n" + "=" * 60)
    print("  Financial Time-Series Regime-Change Experiment")
    print(f"  λ = {lam}  (algorithmic surveillance setting)")
    print("=" * 60)

    # --- Load data ---
    data = None
    if ticker:
        data = load_real_data(ticker, start)

    if data is not None:
        returns, labels, t_star = data
        print(f"  Loaded {ticker}: T={len(returns)}, t*={t_star}")
    else:
        returns, labels, t_star = make_market_data()
        T = len(returns)
        print(f"  Using synthetic market data: T={T}, t*={t_star}")

    # --- Compute detector probabilities ---
    P_vol = volatility_detector(returns, window=20)
    P_cusum = cusum_detector(returns)

    # Smooth for cleaner comparison
    P_vol_smooth = smooth_probabilities(P_vol, method="exponential", alpha=0.3)
    P_cusum_smooth = smooth_probabilities(P_cusum, method="exponential", alpha=0.3)

    detectors = {
        "Volatility detector": P_vol_smooth,
        "CUSUM detector": P_cusum_smooth,
    }

    T = len(returns)
    results = compare_detectors(detectors, t_star=t_star, T=T, lam=lam)

    # --- Print results ---
    print(f"\n  {'Detector':<24} {'AUC':>8} {'HED':>10}")
    print("  " + "-" * 46)
    for name, scores in results.items():
        print(f"  {name:<24} {scores['AUC']:>8.4f} {scores['HED']:>10.4f}")

    print()
    print("  HED captures which detector raised the alarm earliest,")
    print("  directly quantifying the value of earlier warnings.")
    print()

    return {"detectors": detectors, "results": results, "t_star": t_star}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default=None,
                        help="Stock ticker (requires yfinance)")
    parser.add_argument("--start", type=str, default="2007-01-01",
                        help="Start date for real data")
    parser.add_argument("--lam", type=float, default=0.2,
                        help="Hiremath Decay Constant")
    args = parser.parse_args()
    run_financial_experiment(ticker=args.ticker, start=args.start, lam=args.lam)
