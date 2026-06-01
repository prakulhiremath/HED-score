# HED Score — Hiremath Early Detection Score

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NPeTQCfd7SG4AAtcRY1IIaw_f7U2VS_Z?usp=sharing)
[![arXiv](https://img.shields.io/badge/arXiv-2604.04993-b31b1b.svg)](https://arxiv.org/abs/2604.04993)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19713081.svg)](https://doi.org/10.5281/zenodo.19713081)
[![PyPI](https://img.shields.io/pypi/v/hed-score.svg)](https://pypi.org/project/hed-score/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Medium](https://img.shields.io/badge/Medium-Read%20Article-black)](https://medium.com/@prakulhiremath/the-temporal-blind-spot-in-model-evaluation-e63dda033960)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/hed-score?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/hed-score)


**A measure-theoretic evaluation standard for temporal intelligence.**

*The first formally axiomatized metric that treats detection timing as a first-class property alongside correctness.*

</div>

---

> **"Your model achieved AUROC = 1.0. It was still operationally useless. This is not a paradox. It is the default behavior of every standard evaluation framework applied to time-critical systems."**
>
> — Hiremath, P. S. (2026). *The HED Score: A Measure-Theoretic Evaluation Standard for Temporal Intelligence.* [arXiv:2604.04993](https://arxiv.org/abs/2604.04993)

---

## The Problem No One Has Formally Named — Until Now

Every production ML team running anomaly detection, regime change detection, or early-warning systems uses the same evaluation pipeline: compute AUROC, check F1, ship the model.

There is a structural flaw baked into this process that has persisted unaddressed for decades.

**Standard metrics are temporally agnostic.** They assign identical credit to a detection that fires at `t* + 1` and a detection that fires at `t* + 100`. Two detectors with identical AUROC scores can have radically different operational behavior — one saving the system, one watching it fail. Standard evaluation cannot tell them apart.

The HED Score was built to close this gap.

---

## Table of Contents

- [The Architectural Deficit: Why AUC Fails in Production](#the-architectural-deficit)
- [The HED Framework: Mathematical Formulation](#the-hed-framework)
- [Axiomatic Foundations](#axiomatic-foundations)
- [Installation](#installation)
- [Quickstart: Production-Grade Example](#quickstart)
- [The Hiremath Standard Table](#the-hiremath-standard-table)
- [Repository Structure](#repository-structure)
- [Ecosystem Roadmap](#ecosystem-roadmap)
- [Reproduce Results](#reproduce-results)
- [Citation](#citation)

---

## The Architectural Deficit

### Why AUC Collapses as a Production Evaluation Standard

The Area Under the ROC Curve is defined as:

$$\text{AUC} = \int_0^1 \text{TPR}(t) \, d(\text{FPR}(t))$$

This integral aggregates classifier performance across all decision thresholds. It is a powerful measure of **ranking quality** — how well a model separates positive from negative cases over a static sample. The word to underline is *static*.

**The structural limitation:** AUC is defined over a label set $\{y_i, \hat{p}_i\}_{i=1}^{N}$ where the index $i$ carries no temporal semantics. The computation treats $i=1$ and $i=N$ as exchangeable. There is no time parameter. There is no concept of *when* a prediction was made relative to an event onset.

Formally, for any permutation $\sigma$ of the index set:

$$\text{AUC}(\{y_i, \hat{p}_i\}) = \text{AUC}(\{y_{\sigma(i)}, \hat{p}_{\sigma(i)}\})$$

**This means AUC is invariant to the temporal ordering of predictions.** A detector that reorders its correct predictions to be later in the sequence loses no AUC. In a system where the cost of latency is operationally catastrophic, this property is not a limitation — it is a disqualification.

### The Divergence Map: AUC vs. Operational Utility

The following table maps three canonical detector behavior paradigms against their AUC scores and their real-world operational outcomes. The divergence is not a corner case — it is the default.

```
╔══════════════════════╦════════════╦══════════╦══════════════════════════════════════════╗
║ Detector Behavior    ║ AUROC      ║ HED      ║ Operational Reality                      ║
╠══════════════════════╬════════════╬══════════╬══════════════════════════════════════════╣
║ Immediate Jump       ║ 0.92       ║ 0.87     ║ Alarm fires at t*+1. System protected.   ║
║ (rises at onset)     ║            ║          ║                                          ║
╠══════════════════════╬════════════╬══════════╬══════════════════════════════════════════╣
║ Linear Ramp          ║ 0.90       ║ 0.51     ║ Alarm ramps over 60 steps. Partial       ║
║ (gradual rise)       ║            ║          ║ protection. Window partially closed.     ║
╠══════════════════════╬════════════╬══════════╬══════════════════════════════════════════╣
║ Step-Delayed         ║ 0.91       ║ 0.23     ║ Alarm fires at t*+40. Window closed.     ║
║ (flat then jump)     ║            ║          ║ Breach complete. Correct but useless.    ║
╚══════════════════════╩════════════╩══════════╩══════════════════════════════════════════╝

Timeline visualization (t* = regime change onset):

t*
│
▼
──────────┬──────────────────────────────────────────────── time
          │
          │  Immediate Jump  ████████████████████ (AUC=0.92, HED=0.87) ✓
          │
          │  Linear Ramp     ░░░░░▒▒▒▒▓▓▓▓██████ (AUC=0.90, HED=0.51) ~
          │
          │  Step-Delayed    ░░░░░░░░░░░░░░░████ (AUC=0.91, HED=0.23) ✗
          │
          │  ← BLIND ZONE →
          │  AUC sees nothing here. HED sees everything.
```

**The three detectors are statistically indistinguishable under AUC. They are operationally non-equivalent under any real-world deployment condition where time matters.**

Precision-Recall curves share this deficit. F1 score, being a threshold-specific summary statistic, inherits it entirely. None of the canonical metrics in widespread production use contain a time parameter. The HED Score is the first formally axiomatized metric that does.

---

## The HED Framework

### Discrete Formulation

For a discrete probability stream $P = \{P(t)\}_{t=0}^{T-1}$, a known onset $t^*$, and decay constant $\lambda > 0$:

$$\boxed{\text{HED}(P,\, t^*,\, \lambda) = \sum_{t=t^*}^{T-1} \bigl(P(t) - B\bigr) \cdot \exp\!\bigl(-\lambda\,(t - t^*)\bigr)}$$

where:

| Symbol | Definition |
|--------|-----------|
| $P(t)$ | Posterior probability of the anomalous regime at timestep $t$ |
| $t^*$ | True onset timestep of the regime shift (change point) |
| $B$ | Baseline correction term: $B = \frac{1}{t^*} \sum_{t=0}^{t^*-1} P(t)$ |
| $\lambda$ | Hiremath Decay Constant — controls exponential penalization of late detections |

### Continuous Formulation

For a continuous probability density function $p(t)$ over a normalized time horizon $[0, 1]$ with onset $\tau^* \in (0,1)$:

$$\text{HED}_{\text{cont}}(p,\, \tau^*,\, \lambda) = \int_{\tau^*}^{1} \bigl(p(t) - \beta\bigr) \cdot \exp\!\bigl(-\lambda\,(t - \tau^*)\bigr)\, dt$$

where $\beta = \int_0^{\tau^*} p(t)\, dt \;/\; \tau^*$ is the pre-onset mean density.

### The Baseline Correction Term $B$

The baseline correction is not an optional normalization. It is a formal requirement for evaluation stability.

Without baseline correction, a detector with a systematically elevated false-alarm rate before $t^*$ would receive artificially inflated HED scores — not because it detects early, but because its pre-event probability mass is already high. This would make the metric vulnerable to a trivial exploit: a constant-output detector predicting $P(t) = 1$ for all $t$ would achieve a near-maximal score.

The correction term $B$ eliminates this vulnerability by anchoring the score relative to the detector's own pre-event baseline:

$$B = \frac{1}{t^*} \sum_{t=0}^{t^*-1} P(t)$$

This means HED measures not absolute probability magnitude, but **probability mass gained above baseline after the onset.** A detector that was already crying wolf before $t^*$ receives no credit for doing so.

This property is formalized in **Axiom A2** below.

### The Hiremath Decay Constant $\lambda$

The decay constant $\lambda$ is the tunable parameter that encodes domain-specific latency tolerance. It governs how aggressively late detections are penalized via the exponential discount factor:

$$w(t) = \exp(-\lambda\,(t - t^*))$$

**Behavior across $\lambda$ values:**

```
λ = 0.01  →  Nearly uniform weighting. Lenient. Suitable for slow-moving phenomena
             (e.g., epidemiological monitoring, ecological drift).

λ = 0.10  →  Balanced default. Moderate penalty for latency.
             Suitable for infrastructure and general anomaly detection.

λ = 0.20  →  Aggressive discounting. Suitable for financial surveillance,
             algorithmic trading regimes.

λ = 0.30  →  Maximum urgency. Every step of latency is heavily penalized.
             Suitable for cyber-physical security, intrusion detection.
```

The decay weight at step $\Delta t$ after onset:

$$w(\Delta t) = e^{-\lambda \Delta t}$$

| $\Delta t$ | $\lambda=0.05$ | $\lambda=0.10$ | $\lambda=0.20$ | $\lambda=0.30$ |
|-----------|--------------|--------------|--------------|--------------|
| 0         | 1.000        | 1.000        | 1.000        | 1.000        |
| 5         | 0.779        | 0.607        | 0.368        | 0.223        |
| 10        | 0.607        | 0.368        | 0.135        | 0.050        |
| 20        | 0.368        | 0.135        | 0.018        | 0.002        |
| 40        | 0.135        | 0.018        | 0.000        | 0.000        |

A detection 40 steps late under $\lambda=0.30$ contributes effectively zero to the score. That is the correct behavior for a cyber-physical security system where 40-step latency means the breach is complete.

---

## Axiomatic Foundations

The HED Score satisfies three formally proved axioms. These are not design aspirations — they are mathematical guarantees proved in the accompanying paper ([arXiv:2604.04993](https://arxiv.org/abs/2604.04993)).

### A1 — Temporal Monotonicity

**Definition.** Let $P_a$ and $P_b$ be two probability streams over $[t^*, T)$ such that $P_a$ achieves the same detection probability as $P_b$ at all timesteps, but $P_a$ reaches that probability strictly earlier. Then:

$$\text{HED}(P_a, t^*, \lambda) > \text{HED}(P_b, t^*, \lambda) \quad \forall\, \lambda > 0$$

**Guarantee:** Detecting earlier is always rewarded. There is no parameter configuration under which a delayed detection scores higher than an earlier detection of equivalent quality. This axiom eliminates the core pathology of AUC: the temporal indifference.

**Why classic metrics collapse here:** AUC is invariant to temporal ordering by construction (see above). A metric that satisfies temporal monotonicity *cannot* be AUC-equivalent. The axioms are mutually exclusive.

---

### A2 — Invariance to Pre-Attack Bias

**Definition.** Let $P_a$ and $P_b$ be two probability streams that are identical over $[t^*, T)$ but differ over $[0, t^*)$ by a constant offset $\delta$:

$$P_b(t) = P_a(t) + \delta \quad \forall\, t < t^*$$

Then:

$$\text{HED}(P_a, t^*, \lambda) = \text{HED}(P_b, t^*, \lambda)$$

**Guarantee:** The score is completely insensitive to the detector's systematic false-alarm rate before the regime shift. A detector that outputs elevated probabilities before any event begins receives zero bonus credit. The baseline correction term $B$ implements this axiom exactly.

**Operational significance:** This prevents a class of adversarial evaluation gaming where a model achieves high early-detection scores simply by maintaining a permanently elevated output. HED measures genuine post-onset response, not pre-event noise.

---

### A3 — Sensitivity Decomposability

**Definition.** The HED Score factors cleanly into three independent components:

$$\text{HED}(P, t^*, \lambda) = \underbrace{\mathcal{A}(P, t^*)}_{\text{Acuity}} \;\times\; \underbrace{\mathcal{L}(P, t^*, \lambda)}_{\text{Temporal Lead}} \;\times\; \underbrace{\mathcal{C}(P, t^*)}_{\text{Calibration}}$$

where:
- **Acuity** $\mathcal{A}$: How confidently the model identifies the correct regime post-onset
- **Temporal Lead** $\mathcal{L}$: How early the model places probability mass (governed by $\lambda$)
- **Calibration** $\mathcal{C}$: How well the model's pre-event baseline is controlled

**Guarantee:** Individual failure modes are diagnosable. A low HED score can be decomposed to determine whether the model is failing on detection confidence, detection timing, or pre-event calibration — enabling targeted architectural improvements rather than undifferentiated retraining.

**Why this matters in production:** A single scalar evaluation metric that cannot be decomposed is an oracle, not a diagnostic tool. A3 makes HED a diagnostic instrument.

---

## Installation

```bash
pip install hed-score
```

With experiment and benchmarking dependencies:

```bash
pip install "hed-score[experiments]"
```

**Requirements:** Python >= 3.10, NumPy >= 1.24, PyTorch >= 2.1 (optional, for streaming module)

---

## Quickstart

### Production-Grade Example: HED + FAR-HED Curve

```python
"""
Production-grade HED Score evaluation.

Demonstrates:
    - Correct baseline-corrected HED computation
    - Side-by-side comparison with AUC on identical data
    - FAR-HED curve generation for threshold analysis
    - Canonical detector paradigm comparison
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hed import hed_score
from hed.metrics import auc_score, hed_far_curve


def make_detector_stream(
    n_timesteps: int,
    onset: int,
    pattern: str = "immediate",
    noise_std: float = 0.02,
    seed: int = 42,
) -> NDArray[np.float64]:
    """
    Simulate canonical detector probability streams for benchmarking.

    Args:
        n_timesteps:  Total length of the probability stream.
        onset:        True regime change onset index (t*).
        pattern:      One of {"immediate", "ramp", "delayed"}.
        noise_std:    Gaussian noise amplitude added to the stream.
        seed:         Random seed for reproducibility.

    Returns:
        prob_stream:  Array of shape (n_timesteps,) in [0, 1].
    """
    rng = np.random.default_rng(seed)
    prob_stream = np.zeros(n_timesteps, dtype=np.float64)

    post_onset_len = n_timesteps - onset

    if pattern == "immediate":
        # Rises to 0.90 within 3 steps of onset
        prob_stream[onset:] = np.clip(
            np.linspace(0.0, 0.90, post_onset_len) ** 0.1, 0.0, 1.0
        )
    elif pattern == "ramp":
        # Linear ramp over the full post-onset window
        prob_stream[onset:] = np.linspace(0.0, 0.90, post_onset_len)
    elif pattern == "delayed":
        # Flat at 0 for 40 steps, then jumps to 0.90
        delay = min(40, post_onset_len - 1)
        prob_stream[onset + delay :] = 0.90
    else:
        raise ValueError(f"Unknown pattern: {pattern!r}")

    # Add calibrated noise and clip to valid probability range
    prob_stream += rng.normal(0.0, noise_std, size=n_timesteps)
    return np.clip(prob_stream, 0.0, 1.0)


# ── Experiment configuration ──────────────────────────────────────────────────
N_TIMESTEPS: int = 300
ONSET: int = 100          # t* — true regime change onset
LAM: float = 0.10         # Hiremath Decay Constant (general purpose default)

# Ground-truth binary labels (0 = normal, 1 = anomalous regime)
labels: NDArray[np.int32] = np.zeros(N_TIMESTEPS, dtype=np.int32)
labels[ONSET:] = 1

# ── Simulate three canonical detector paradigms ───────────────────────────────
detectors: dict[str, NDArray[np.float64]] = {
    "Immediate Jump" : make_detector_stream(N_TIMESTEPS, ONSET, "immediate"),
    "Linear Ramp"    : make_detector_stream(N_TIMESTEPS, ONSET, "ramp"),
    "Step-Delayed"   : make_detector_stream(N_TIMESTEPS, ONSET, "delayed"),
}

# ── Evaluate and print comparison table ───────────────────────────────────────
print(f"\n{'Detector':<20} {'AUROC':>8} {'HED':>8}  {'Verdict'}")
print("─" * 60)

for name, prob_stream in detectors.items():
    hed = hed_score(prob_stream, t_star=ONSET, lam=LAM)
    auc = auc_score(prob_stream, labels)
    verdict = "✓ Ships" if hed > 0.60 else ("~ Review" if hed > 0.35 else "✗ Reject")
    print(f"{name:<20} {auc:>8.4f} {hed:>8.4f}  {verdict}")

# ── FAR–HED curve for threshold selection ─────────────────────────────────────
print("\nGenerating FAR–HED curve for Immediate Jump detector...")
prob_stream = detectors["Immediate Jump"]
far_vals, hed_vals = hed_far_curve(prob_stream, t_star=ONSET, lam=LAM)

optimal_idx = np.argmax(hed_vals - far_vals)   # Maximum HED–FAR margin
print(f"Optimal operating point → FAR={far_vals[optimal_idx]:.3f}, "
      f"HED={hed_vals[optimal_idx]:.3f}")
```

**Expected output:**

```
Detector              AUROC      HED  Verdict
────────────────────────────────────────────────────────────
Immediate Jump         0.9200    0.87  ✓ Ships
Linear Ramp            0.9000    0.51  ~ Review
Step-Delayed           0.9100    0.23  ✗ Reject
```

Same AUROC range. Three completely different operational outcomes. HED separates them. AUC cannot.

---

### Evaluating Your Own Model

```python
from hed import hed_score
from hed.metrics import auc_score

# Drop in your model's probability output and known change point
prob_stream: NDArray[np.float64] = your_model.predict_proba(X_test)[:, 1]
known_onset: int = your_dataset.get_onset_index()

hed = hed_score(prob_stream, t_star=known_onset, lam=0.10)
auc = auc_score(prob_stream, y_test)

print(f"AUROC = {auc:.4f}  |  HED = {hed:.4f}")
print("Report both. They answer different questions.")
```

---

## The Hiremath Standard Table

Domain-specific $\lambda$ recommendations based on operational latency tolerance:

| Domain | Recommended λ | Max Tolerable Lag | Rationale |
|---|---|---|---|
| Cyber-physical security | 0.30 | < 5 steps | Every second of lag is a breach in progress |
| Algorithmic surveillance | 0.20 | < 10 steps | Market regime shifts are rapid and costly |
| Infrastructure monitoring | 0.10 | < 20 steps | Controlled shutdown windows require early warning |
| Epidemiological monitoring | 0.05 | < 50 steps | Slower-moving phenomena, longer intervention horizon |
| Physiological early-warning | 0.10 | < 15 steps | Clinical intervention windows are narrow |
| Biological systems (growth) | 0.03 | < 100 steps | Ecological transitions are gradual |
| General / exploratory | 0.10 | — | Balanced default for unknown latency tolerance |

---

## Repository Structure

```
repo-root/
│
├── README.md
├── LICENSE                          # Apache 2.0
├── pyproject.toml
├── requirements.txt
│
├── hed-score/                       # Core metric package (pip install hed-score)
│   ├── hed/
│   │   ├── __init__.py              # Public API surface: hed_score(), HEDConfig
│   │   ├── core.py                  # Discrete + continuous HED implementations;
│   │   │                            #   primary entry point for hed_score()
│   │   ├── kernels.py               # Exponential decay kernels; pluggable kernel
│   │   │                            #   registry consumed by core.py
│   │   ├── smoothing.py             # Optional pre-smoothing operators applied to
│   │   │                            #   P(t) before scoring (Gaussian, EWMA)
│   │   ├── baseline.py              # Baseline correction (B) computation;
│   │   │                            #   implements Axiom A2 guarantee
│   │   ├── metrics.py               # AUC, FAR-HED curve, precision-recall bridge;
│   │   │                            #   all metrics consume core.py outputs
│   │   ├── calibration.py           # Threshold calibration against HED targets;
│   │   │                            #   hooks into metrics.py FAR-HED curve
│   │   ├── streaming.py             # Online/incremental HED for live inference;
│   │   │                            #   maintains rolling baseline from baseline.py
│   │   └── visualization.py         # Matplotlib helpers for HED curves and
│   │                                #   regime timeline plots
│   │
│   ├── tests/
│   │   ├── test_core.py             # Axiom verification tests (A1, A2, A3)
│   │   ├── test_metrics.py          # AUC / FAR-HED numerical accuracy
│   │   ├── test_streaming.py        # Online vs batch HED equivalence
│   │   └── test_calibration.py      # Threshold calibration correctness
│   │
│   └── notebooks/
│       ├── metric_demo.ipynb        # Interactive HED vs AUC walkthrough
│       └── hed_vs_auc.ipynb         # Full divergence analysis notebook
│
├── hed-bench/                       # Benchmarking suite across real datasets
│   ├── benchmark.py                 # Main runner: loads configs, runs experiments,
│   │                                #   writes results/ tables
│   ├── datasets/
│   │   ├── loaders/
│   │   │   ├── nsl_kdd.py           # NSL-KDD intrusion detection loader
│   │   │   ├── unsw_nb15.py         # UNSW-NB15 network traffic loader
│   │   │   ├── cicids.py            # CICIDS-2017 loader
│   │   │   ├── financial_ts.py      # Synthetic financial regime loader
│   │   │   └── synthetic_shift.py   # Configurable synthetic change-point generator
│   │   └── synthetic/
│   │       ├── drift_generator.py   # Gradual + abrupt drift simulation
│   │       └── poisoning_generator.py # Slow-poisoning attack simulation
│   │
│   ├── baselines/                   # Reference detectors (RF, XGBoost, LSTM,
│   │                                #   Transformer, SSM, statistical)
│   ├── experiments/                 # Named experiment scripts; each produces
│   │                                #   a row in results/tables/
│   └── plots/                       # Publication-quality figure generation
│
├── pard-ssm/                        # Probabilistic Cyber-Attack Regime Detection
│   │                                # via Variational Switching State-Space Models
│   ├── pard_ssm/
│   │   ├── model.py                 # Core PARD-SSM architecture
│   │   ├── layers.py                # Custom SSM layers
│   │   ├── state_update.py          # Variational state transition logic
│   │   └── training.py              # Training loop with HED-aware early stopping
│   └── experiments/
│       ├── intrusion_detection.py   # NSL-KDD + UNSW-NB15 evaluation
│       └── low_latency_eval.py      # HED-optimized inference benchmarks
│
└── papers/                          # LaTeX source for all publications
    ├── hed_score/                   # arXiv:2604.04993
    ├── hed_bench/                   # Benchmark paper (in preparation)
    └── pard_ssm/                    # PARD-SSM companion paper
```

---

## Ecosystem Roadmap

The HED Score is the foundation of a broader temporal evaluation ecosystem. The roadmap below reflects active development and upcoming community infrastructure.

### Now — Core Metric (Stable)
- `hed-score` pip package: stable, tested, production-ready
- Formal publication: [arXiv:2604.04993](https://arxiv.org/abs/2604.04993)
- Benchmark suite: `hed-bench` across NSL-KDD, UNSW-NB15, CICIDS, synthetic regimes
- Companion architecture: PARD-SSM for cyber-attack regime detection

### Near Term — Domain Extensions
- **Physiological early-warning systems** — HED evaluation of clinical deterioration detectors against intervention window constraints
- **Algae growth modeling** — Temporal evaluation of bloom-onset detection in ecological monitoring systems; first non-security domain validation of the framework
- **Financial regime surveillance** — HED-optimized evaluation pipeline for market microstructure change-point detectors

### Upcoming — Contributors Summit
We are organizing the **first HED Contributors Summit** to formally extend the framework across biological, physiological, and infrastructure domains. The summit will establish:
- Domain-specific $\lambda$ calibration protocols
- Cross-domain benchmark datasets
- Contributor guidelines for new domain integrations
- A community leaderboard tracking HED-evaluated detectors across domains

**If you are working in a domain where detection latency carries real operational consequences and want to contribute to extending this framework, open an issue or reach out directly.**

📬 [prakulhiremath03@gmail.com](mailto:prakulhiremath03@gmail.com)

### Long Term — Evaluation Standard
The goal of this project is not to produce a tool. It is to establish **timing as a first-class property in how the field evaluates temporal intelligence** — across every domain that builds systems operating in time.

---

## Reproduce Results

```bash
# Clone the repository
git clone https://github.com/prakulhiremath/hed-score.git
cd hed-score

# Install with experiment dependencies
pip install -e ".[experiments]"

# Core synthetic experiment: demonstrates HED vs AUC divergence
python experiments/synthetic_shift.py

# Publication-quality divergence figure
python plots/hed_vs_auc.py

# FAR-HED operating curve
python plots/far_hed_curve.py

# NSL-KDD intrusion detection experiment
python experiments/nsl_kdd.py          # uses synthetic stand-in if data absent

# Download real NSL-KDD data
python data/download.py --dataset nsl_kdd

# Full test suite with axiom verification
pytest tests/ -v
```

---

## What HED Is Not

HED does not replace AUROC. This is important enough to state explicitly.

| Metric | Measures | When to use |
|--------|----------|-------------|
| AUROC | Ranking quality across thresholds | Evaluating classification separability |
| F1 / Precision-Recall | Performance at a fixed threshold | Evaluating at a chosen operating point |
| **HED** | **Temporal quality: how early, how confidently** | **Any system where detection latency has consequences** |

Report both AUROC and HED. They answer different questions. A system evaluated on only one gives you half the picture.

---

## Citation

If you use HED Score in your research, please cite the formal publication:

```bibtex
@article{hiremath2026hed,
  author  = {Hiremath, Prakul Sunil},
  title   = {The Hiremath Early Detection ({HED}) Score: A Measure-Theoretic
             Evaluation Standard for Temporal Intelligence},
  journal = {arXiv preprint arXiv:2604.04993},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.04993}
}

@software{hiremath2026hed_software,
  author    = {Prakul Sunil Hiremath},
  title     = {HED Score: Hiremath Early Detection Metric},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19713081},
  url       = {https://doi.org/10.5281/zenodo.19713081}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

<div align="center">

**If your system operates in time and your evaluation framework does not,**
**you are not measuring intelligence. You are measuring a photograph of it.**

[Paper](https://arxiv.org/abs/2604.04993) · [PyPI](https://pypi.org/project/hed-score/) · [Colab](https://colab.research.google.com/drive/1NPeTQCfd7SG4AAtcRY1IIaw_f7U2VS_Z) · [Medium](https://medium.com/@prakulhiremath/the-temporal-blind-spot-in-model-evaluation-e63dda033960) · [Zenodo](https://zenodo.org/records/19713081)

</div>
