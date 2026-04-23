"""
experiments/synthetic_shift.py
-------------------------------
Key demonstration from the HED paper.

Shows two detectors with similar AUC but very different HED scores,
illustrating why AUC alone is insufficient for evaluating early
detection systems.

Run:
    python experiments/synthetic_shift.py

Output:
    Console table + saved figure: plots/synthetic_hed_vs_auc.png
"""

import numpy as np
import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hed import hed_score
from hed.metrics import auc_score, compare_detectors
from hed.utils import make_step_detector, make_ramp_detector, make_delayed_detector


def run_synthetic_experiment(
    T: int = 300,
    t_star: int = 100,
    lam: float = 0.1,
    seed: int = 42,
    save_fig: bool = True,
    verbose: bool = True,
) -> dict:
    """Run the synthetic regime-shift experiment.

    Creates three detectors:
      1. Fast detector  : jumps to high probability immediately at t_star.
      2. Ramp detector  : slowly increases probability after t_star.
      3. Delayed detector: fires 40 steps after t_star.

    All three have similar AUC (they all distinguish anomalous from
    normal eventually), but HED clearly separates them by temporal quality.

    Parameters
    ----------
    T : int
        Total time steps.
    t_star : int
        True regime-shift onset index.
    lam : float
        Hiremath Decay Constant.
    seed : int
        Base random seed.
    save_fig : bool
        If True, save probability-stream figure to plots/.
    verbose : bool
        If True, print results table to stdout.

    Returns
    -------
    dict
        {'detectors': {...}, 'results': {...}}
    """
    rng_fast = np.random.default_rng(seed)
    rng_ramp = np.random.default_rng(seed + 1)
    rng_delay = np.random.default_rng(seed + 2)

    P_fast = make_step_detector(T, t_star, noise=0.04, rng=rng_fast)
    P_ramp = make_ramp_detector(T, t_star, noise=0.04, rng=rng_ramp)
    P_delayed = make_delayed_detector(T, t_star, delay=40, noise=0.04, rng=rng_delay)

    detectors = {
        "Fast detector": P_fast,
        "Ramp detector": P_ramp,
        "Delayed detector": P_delayed,
    }

    results = compare_detectors(detectors, t_star=t_star, T=T, lam=lam)

    if verbose:
        _print_table(results, t_star, T, lam)

    if save_fig:
        _save_figure(detectors, results, t_star, T, lam)

    return {"detectors": detectors, "results": results}


def _print_table(results: dict, t_star: int, T: int, lam: float) -> None:
    """Pretty-print the comparison table."""
    print("\n" + "=" * 60)
    print("  Synthetic Regime-Shift Experiment")
    print(f"  T={T}, t*={t_star}, λ={lam}")
    print("=" * 60)
    print(f"  {'Detector':<22} {'AUC':>8} {'HED':>10}")
    print("  " + "-" * 44)
    for name, scores in results.items():
        print(f"  {name:<22} {scores['AUC']:>8.4f} {scores['HED']:>10.4f}")
    print("=" * 60)
    print()
    print("  Interpretation:")
    print("  - AUC values are similar across all detectors.")
    print("  - HED clearly separates fast from slow/delayed detection.")
    print("  - This is the key advantage of HED over AUC for time-critical tasks.")
    print()


def _save_figure(
    detectors: dict,
    results: dict,
    t_star: int,
    T: int,
    lam: float,
) -> None:
    """Save probability stream figure (delegates to plots/hed_vs_auc.py)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs("plots", exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#2196F3", "#FF9800", "#E91E63"]
        t = np.arange(T)

        for (name, P), color in zip(detectors.items(), colors):
            scores = results[name]
            label = f"{name}  |  AUC={scores['AUC']:.3f}, HED={scores['HED']:.3f}"
            ax.plot(t, P, color=color, lw=1.8, alpha=0.85, label=label)

        ax.axvline(t_star, color="black", ls="--", lw=1.5, label=f"Regime shift (t*={t_star})")
        ax.fill_between(t[:t_star], 0, 1, alpha=0.06, color="green", label="Pre-shift (normal)")
        ax.fill_between(t[t_star:], 0, 1, alpha=0.06, color="red", label="Post-shift (anomalous)")

        ax.set_xlabel("Time step", fontsize=12)
        ax.set_ylabel("Posterior probability P(anomalous | data)", fontsize=12)
        ax.set_title(
            f"HED reveals temporal quality that AUC misses  (λ={lam})",
            fontsize=13, fontweight="bold"
        )
        ax.legend(fontsize=9, loc="upper left")
        ax.set_xlim(0, T)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        path = "plots/synthetic_hed_vs_auc.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure saved → {path}")

    except ImportError:
        print("  [Note] matplotlib not installed; figure not saved.")


if __name__ == "__main__":
    run_synthetic_experiment()
