"""
plots/far_hed_curve.py
-----------------------
FAR–HED curve: the temporal analogue of the ROC curve.

The x-axis is the False Alarm Rate (FAR).
The y-axis is the HED Score at each decision threshold.

A perfect early detector hugs the upper-left corner:
  - Low FAR  (few false alarms before the shift)
  - High HED (probability mass concentrated early after the shift)

Contrast with the ROC curve, which ignores *when* a detection occurs.

Run:
    python plots/far_hed_curve.py
    python plots/far_hed_curve.py --lam 0.3 --out figures/far_hed.pdf
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hed.metrics import hed_far_curve
from hed.utils import make_step_detector, make_ramp_detector, make_delayed_detector


def plot_far_hed_curve(
    T: int = 300,
    t_star: int = 100,
    lam: float = 0.1,
    n_thresholds: int = 200,
    out_path: str = "plots/far_hed_curve.png",
    dpi: int = 180,
) -> None:
    """Generate the FAR–HED curve figure.

    Parameters
    ----------
    T : int
    t_star : int
    lam : float
    n_thresholds : int
        Number of thresholds to sweep.
    out_path : str
    dpi : int
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib")
        return

    # ---- Detectors ----
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(1)
    rng_c = np.random.default_rng(2)

    detectors = [
        ("Fast detector",    make_step_detector(T, t_star, rng=rng_a),    "#1565C0"),
        ("Ramp detector",    make_ramp_detector(T, t_star, rng=rng_b),    "#E65100"),
        ("Delayed detector", make_delayed_detector(T, t_star, delay=50, rng=rng_c), "#AD1457"),
    ]

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, P, color in detectors:
        far, hed = hed_far_curve(P, t_star, lam=lam, n_thresholds=n_thresholds)
        # Sort by FAR for clean line
        order = np.argsort(far)
        ax.plot(far[order], hed[order], color=color, lw=2.0, label=name)

    # Random baseline
    ax.plot([0, 1], [0, 0], "k--", lw=1, alpha=0.5, label="Baseline (no detection)")

    # Ideal point
    ax.scatter([0], [1], color="gold", s=120, zorder=5,
               edgecolor="black", linewidth=0.8, label="Ideal (FAR=0, HED=1)")

    ax.set_xlabel("False Alarm Rate (FAR)", fontsize=12)
    ax.set_ylabel(f"HED Score  (λ={lam})", fontsize=12)
    ax.set_title(
        "FAR–HED Curve\n"
        "Upper-left = better temporal detection at low false-alarm cost",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.25)

    # AUC-HED (area under this curve)
    for name, P, color in detectors:
        far, hed = hed_far_curve(P, t_star, lam=lam, n_thresholds=n_thresholds)
        order = np.argsort(far)
        area = float(np.trapezoid(hed[order], far[order]))
        print(f"  {name:<22}  AUHC (Area Under HED Curve) = {area:.4f}")

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--t_star", type=int, default=100)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--n_thresholds", type=int, default=200)
    parser.add_argument("--out", type=str, default="plots/far_hed_curve.png")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    plot_far_hed_curve(
        T=args.T,
        t_star=args.t_star,
        lam=args.lam,
        n_thresholds=args.n_thresholds,
        out_path=args.out,
        dpi=args.dpi,
    )
