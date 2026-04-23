# HED Score

[![arXiv](https://img.shields.io/badge/arXiv-2604.04993-b31b1b.svg)](https://arxiv.org/abs/2604.04993)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19713081.svg)](https://doi.org/10.5281/zenodo.19713081)
[![PyPI version](https://img.shields.io/pypi/v/hed-score.svg)](https://pypi.org/project/hed-score/)
[![Python](https://img.shields.io/pypi/pyversions/hed-score.svg)](https://pypi.org/project/hed-score/)

**Hiremath Early Detection (HED) Score** — a measure-theoretic metric for evaluating early detection in time-series anomaly and regime-change tasks.

> Hiremath, P. S. (2026). *The Hiremath Early Detection (HED) Score: A Measure-Theoretic Evaluation Standard for Temporal Intelligence.* arXiv:2604.04993 [stat.ML]

---

## What is the HED Score?

HED quantifies **how early** a detector raises its probability mass after a regime shift.
It integrates the baseline-corrected posterior probability stream through an exponential decay kernel:

```
HED(P, t*, λ) = Σ_{t=t*}^{T-1}  (P(t) − B) · exp(−λ (t − t*))
```

where:
- `P(t)` — posterior probability of the anomalous regime at time `t`
- `t*`   — true onset of the regime shift
- `B`    — mean of `P` before `t*` (baseline correction)
- `λ`    — Hiremath Decay Constant: controls how steeply late detections are penalised

The resulting scalar simultaneously encodes **detection acuity**, **temporal lead**, and **pre-transition calibration quality**.

## Why does AUC fail?

AUC is **temporally agnostic**: it assigns identical credit to a detection at `t*+1` and a detection at `t*+100`.  Two detectors with identical AUC can have radically different behaviour — one fires immediately after the shift; the other fires 50 steps later.  In time-critical domains (cyber-physical security, epidemic monitoring, financial surveillance), this distinction is everything.

**HED resolves this** by exponentially discounting detections that occur late after the shift.

```
                 AUC      HED
Fast detector    0.92     0.87   ← rewards early rise
Ramp detector    0.90     0.51   ← penalises slow ramp
Delayed detector 0.91     0.23   ← penalises 40-step lag
```

Same AUC. Very different temporal quality.

---

## Installation

```bash
pip install hed-score
```

With experiment dependencies:

```bash
pip install "hed-score[experiments]"
```

---

## Quick example

```python
import numpy as np
from hed import hed_score

T      = 200    # total time steps
t_star = 100    # regime shift begins here
lam    = 0.1    # Hiremath Decay Constant

# A fast detector (jumps to 0.9 immediately)
P_fast = np.zeros(T)
P_fast[t_star:] = 0.9

# A slow detector (ramps up over 100 steps)
P_slow = np.zeros(T)
P_slow[t_star:] = np.linspace(0, 0.9, T - t_star)

print(hed_score(P_fast, t_star, lam=lam))   # → ~0.87
print(hed_score(P_slow, t_star, lam=lam))   # → ~0.45
```

Any model that outputs a probability stream can be evaluated:

```python
from hed import hed_score
from hed.metrics import auc_score, hed_far_curve

# Your model's output
P = your_model.predict_proba(X_test)[:, 1]

# Evaluate
hed = hed_score(P, t_star=known_onset, lam=0.1)
auc = auc_score(P, labels)
print(f"HED={hed:.4f}  AUC={auc:.4f}")

# FAR–HED curve
far, hed_vals = hed_far_curve(P, t_star=known_onset, lam=0.1)
```

---

## Hiremath Standard Table

| Domain | Recommended λ | Rationale |
|---|---|---|
| Cyber-physical security | 0.3 | Every second of early warning is critical |
| Epidemiological monitoring | 0.05 | Slower-moving outbreaks, longer horizon |
| Algorithmic surveillance | 0.2 | Market movements can be rapid |
| General / exploratory | 0.1 | Balanced default |

---

## Reproduce results

```bash
# Clone
git clone https://github.com/prakulhiremath/hed-score.git
cd hed-score

# Install
pip install -e ".[experiments]"

# Key synthetic experiment (demonstrates HED vs AUC)
python experiments/synthetic_shift.py

# Publication-quality figure
python plots/hed_vs_auc.py

# FAR–HED curve
python plots/far_hed_curve.py

# NSL-KDD experiment (synthetic stand-in if data not downloaded)
python experiments/nsl_kdd.py

# Download real NSL-KDD data
python data/download.py --dataset nsl_kdd

# Run tests
pytest tests/ -v
```

---

## Repository structure

```
hed-score/
├── hed/
│   ├── __init__.py       # Public API
│   ├── core.py           # HED implementation (discrete + continuous)
│   ├── utils.py          # Baseline correction, kernel, smoothing
│   └── metrics.py        # AUC, FAR, HED-FAR curve
├── experiments/
│   ├── synthetic_shift.py  # Key demo: same AUC, different HED
│   ├── nsl_kdd.py          # Cyber-intrusion detection
│   └── financial_ts.py     # Market regime change
├── models/
│   ├── rf.py               # Random Forest detector
│   ├── lstm.py             # LSTM detector
│   └── pard_ssm.py         # PARD-SSM placeholder
├── plots/
│   ├── hed_vs_auc.py       # Main comparison figure
│   └── far_hed_curve.py    # FAR–HED curve
├── data/
│   └── download.py         # Dataset download scripts
├── tests/
│   ├── test_core.py
│   └── test_metrics.py
├── notebooks/
│   └── demo.ipynb
├── pyproject.toml
└── requirements.txt
```

---

## Key properties

The HED Score satisfies three axiomatic requirements (proved in the paper):

- **A1 Temporal Monotonicity** — detecting earlier always yields a higher score.
- **A2 Invariance to Pre-Attack Bias** — the score is unaffected by a detector's systematic false-alarm rate before the shift.
- **A3 Sensitivity Decomposability** — the score factors cleanly into acuity × lead × calibration components.

---

## Citation

```bibtex
@software{hiremath2026hed,
  author       = {Prakul Sunil Hiremath},
  title        = {HED Score: Hiremath Early Detection Metric},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19713081},
  url          = {https://doi.org/10.5281/zenodo.19713081}
}
```

---

## License

Apache 2.0
