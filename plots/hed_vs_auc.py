"""
plots/hed_vs_auc.py
--------------------
Publication-quality figure: same AUC, different HED.

This is the most important plot in the package — the central illustration
of why HED is needed.

Produces two panels:
  Left  : Probability streams over time (with regime-shift marker)
  Right : AUC vs HED bar chart comparison

Run:
    python plots/hed_vs_auc.py
    python plots/hed_vs_auc.py --lam 0.3 --out figures/fig1.pdf

Output: plots/hed_vs_auc.png  (or path given by --out)
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hed import hed_score
from hed.metrics import auc_score
from hed.utils import make_step_detector, make_ramp_detector, make_delayed_detector


def plot_hed_vs_auc(
    T: int = 300,
    t_star: int = 100,
    lam: float = 0.1,
    out_path: str = "plots/hed_vs_auc.png",
    dpi: int = 180,
) -> None:
    """Generate the two-panel HED vs AUC comparison figure.

    Parameters
    ----------
    T : int
        Total time steps.
    t_star : int
        Regime-shift onset.
    lam : float
        Hiremath Decay Constant for HED.
    out_path : str
        Output path.  Extension determines format (.png / .pdf / .svg).
    dpi : int
        Resolution for raster formats.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("matplotlib is required for plotting: pip install matplotlib")
        return

    # ---- Reproducible detectors ----
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(1)
    rng_c = np.random.default_rng(2)

    P_fast = make_step_detector(T, t_star, noise=0.035, rng=rng_a)
    P_ramp = make_ramp_detector(T, t_star, noise=0.035, rng=rng_b)
    P_delayed = make_delayed_detector(T, t_star, delay=50, noise=0.035, rng=rng_c)

    # ---- Labels for AUC ----
    labels = np.zeros(T, dtype=int)
    labels[t_star:] = 1

    detectors = [
        ("Fast detector", P_fast, "#1565C0"),     # deep blue
        ("Ramp detector", P_ramp, "#E65100"),      # deep orange
        ("Delayed detector", P_delayed, "#AD1457"), # deep pink
    ]

    scores = []
    for name, P, color in detectors:
        hed = hed_score(P, t_star, lam=lam, normalise=True)
        auc = auc_score(P, labels)
        scores.append((name, hed, auc, color))

    # =====================================================================
    # Layout
    # =====================================================================
    fig = plt.figure(figsize=(13, 5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2, 1], wspace=0.32)
    ax_streams = fig.add_subplot(gs[0, 0])
    ax_bars    = fig.add_subplot(gs[0, 1])

    t = np.arange(T)

    # =====================================================================
    # Panel A — Probability streams
    # =====================================================================
    for name, P, color in detectors:
        ax_streams.plot(t, P, color=color, lw=1.8, alpha=0.88,
                        label=name, zorder=3)

    # Regime-shift vertical line
    ax_streams.axvline(
        t_star, color="#37474F", lw=1.6, ls="--", zorder=4,
        label=f"Regime shift  t* = {t_star}"
    )

    # Shaded regions
    ax_streams.axvspan(0, t_star, alpha=0.07, color="#4CAF50", label="Normal regime")
    ax_streams.axvspan(t_star, T, alpha=0.07, color="#F44336", label="Anomalous regime")

    # Exponential decay overlay (illustrative)
    k = np.arange(T - t_star)
    decay = np.exp(-lam * k)
    ax_streams.fill_between(
        t[t_star:], 0, decay, alpha=0.12, color="#7E57C2",
        label=f"Decay kernel  exp(−{lam}·Δt)", zorder=2
    )

    ax_streams.set_xlabel("Time step  t", fontsize=11)
    ax_streams.set_ylabel("P(anomalous | data)", fontsize=11)
    ax_streams.set_title(
        "(A)  Probability streams — same AUC, different HED",
        fontsize=11, fontweight="bold", pad=10
    )
    ax_streams.set_xlim(0, T)
    ax_streams.set_ylim(-0.05, 1.05)
    ax_streams.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax_streams.grid(True, alpha=0.25)

    # =====================================================================
    # Panel B — Bar chart: HED vs AUC
    # =====================================================================
    n = len(scores)
    x = np.arange(n)
    bar_w = 0.35

    auc_bars = ax_bars.bar(
        x - bar_w / 2,
        [s[2] for s in scores],   # AUC values
        bar_w, label="AUC",
        color=[s[3] for s in scores],
        alpha=0.45, edgecolor="grey", linewidth=0.8,
        hatch="///",
    )
    hed_bars = ax_bars.bar(
        x + bar_w / 2,
        [s[1] for s in scores],   # HED values
        bar_w, label=f"HED  (λ={lam})",
        color=[s[3] for s in scores],
        alpha=0.88, edgecolor="grey", linewidth=0.8,
    )

    # Annotate bar values
    for bar in auc_bars:
        h = bar.get_height()
        ax_bars.text(
            bar.get_x() + bar.get_width() / 2, h + 0.012,
            f"{h:.3f}", ha="center", va="bottom", fontsize=8, color="#555"
        )
    for bar in hed_bars:
        h = bar.get_height()
        ax_bars.text(
            bar.get_x() + bar.get_width() / 2, h + 0.012,
            f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold"
        )

    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(
        [s[0].replace(" detector", "\ndetector") for s in scores],
        fontsize=9
    )
    ax_bars.set_ylim(0, 1.18)
    ax_bars.set_ylabel("Score", fontsize=11)
    ax_bars.set_title(
        "(B)  AUC vs HED comparison",
        fontsize=11, fontweight="bold", pad=10
    )
    ax_bars.legend(fontsize=9)
    ax_bars.grid(True, axis="y", alpha=0.25)

    # Key callout box
    callout = (
        "AUC is nearly identical\n"
        "across all detectors.\n\n"
        "HED clearly separates\n"
        "fast from slow detection."
    )
    ax_bars.text(
        0.97, 0.97, callout,
        transform=ax_bars.transAxes,
        fontsize=8, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4", alpha=0.9,
                  edgecolor="#F9A825"),
    )

    # =====================================================================
    # Global styling
    # =====================================================================
    fig.suptitle(
        "The HED Score captures temporal detection quality that AUC ignores",
        fontsize=12, fontweight="bold", y=1.01
    )

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HED vs AUC figure.")
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--t_star", type=int, default=100)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--out", type=str, default="plots/hed_vs_auc.png")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    plot_hed_vs_auc(
        T=args.T,
        t_star=args.t_star,
        lam=args.lam,
        out_path=args.out,
        dpi=args.dpi,
    )
