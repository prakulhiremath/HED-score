# Contributing to HED Score

First: thank you for being here.

The HED Score exists because a gap in how the field evaluates temporal intelligence was too important to leave unnamed. This project is not a paper that got open-sourced as an afterthought. It is a living framework being extended across domains — cybersecurity, physiological monitoring, financial surveillance, ecological systems — anywhere that a late correct prediction is operationally indistinguishable from a wrong one.

If you are reading this file, you have likely hit that wall yourself. That is exactly who this project is for.

---

## Table of Contents

- [The Philosophy of This Project](#the-philosophy-of-this-project)
- [Where You Can Contribute](#where-you-can-contribute)
- [The Contributors Summit](#the-contributors-summit)
- [Domain Extensions](#domain-extensions)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Code Standards](#code-standards)
- [Adding a New Domain](#adding-a-new-domain)
- [Writing Tests](#writing-tests)
- [Documentation Standards](#documentation-standards)
- [Reporting Issues](#reporting-issues)
- [Code of Conduct](#code-of-conduct)
- [Contact](#contact)

---

## The Philosophy of This Project

Three principles govern every decision in this codebase.

**1. Timing is not a secondary concern.** Every design choice, API decision, and test case exists to serve one thesis: that detection timing deserves to be a first-class property in model evaluation. A contribution that makes HED faster but less temporally precise is not a good contribution. A contribution that makes HED harder to understand is not a good contribution. The metric must remain interpretable to the practitioner who needs it, not just the researcher who proved it.

**2. Axioms are guarantees, not aspirations.** The three core axioms — Temporal Monotonicity (A1), Invariance to Pre-Attack Bias (A2), and Sensitivity Decomposability (A3) — are formally proved properties. Any contribution that modifies `core.py`, `baseline.py`, or `kernels.py` must include a test demonstrating that all three axioms still hold under the change. This is non-negotiable. A metric that sometimes violates Temporal Monotonicity is not HED.

**3. Domain extensions are first-class contributions.** Porting HED to a new domain — validating it, calibrating the Hiremath Decay Constant λ for that domain's operational latency tolerance, and documenting the results — is as valuable as improving the core implementation. The framework only becomes a standard if it survives contact with real problems across multiple fields.

---

## Where You Can Contribute

### High Priority

These are the areas where contributions have the most immediate impact:

**Domain validation and λ calibration**
The most important open problem in this project is rigorous λ calibration across new domains. We have strong defaults for cybersecurity (λ=0.30) and epidemiological monitoring (λ=0.05). We need validated protocols for:
- Physiological early-warning systems (clinical deterioration, sepsis onset)
- Ecological monitoring (algae bloom onset, species population regime shifts)
- Industrial fault detection (manufacturing line anomaly, equipment failure prediction)
- Energy grid anomaly detection

If you work in any of these domains and have access to labeled time-series data with known onset points, your domain expertise is the contribution.

**Streaming evaluation improvements**
`streaming.py` implements online HED computation but currently recomputes baseline estimates on a rolling window. There is a more elegant incremental formulation that maintains exact baseline correction without full recomputation. See [Issue #TBD] for the mathematical sketch.

**Benchmark dataset loaders**
`hed-bench/datasets/loaders/` is the highest-leverage place for new contributors. A well-written loader for a public anomaly detection dataset — with documented preprocessing, onset labeling methodology, and a validation notebook — makes the entire benchmark suite more credible.

**Visualization and interpretability**
`visualization.py` currently produces functional plots. It does not produce publication-quality figures. Contributors who care about communicating results clearly — axis labeling, color schemes that work in grayscale, export pipelines for LaTeX — are welcome here.

### Good First Issues

If you are new to the codebase, start here:

- Add type hints to any function in `metrics.py` that is missing them
- Write a docstring for any public function that lacks one
- Add a test case that verifies Axiom A1 (Temporal Monotonicity) for the continuous HED formulation
- Improve error messages in `core.py` for invalid input shapes
- Add a `--help` description to any experiment script in `hed-bench/experiments/`

---

## The Contributors Summit

We are organizing the first **HED Contributors Summit** to formally extend the evaluation framework across biological, physiological, and infrastructure domains.

The summit will establish:
- Domain-specific λ calibration protocols with formal justification
- Cross-domain benchmark datasets and preprocessing standards
- A community leaderboard tracking HED-evaluated detectors by domain
- Contributor guidelines for integrating new domains into `hed-bench`

**If you want to be involved in the summit, open an issue titled `[Summit] <your domain>` describing the domain you work in, the type of data you have access to, and the operational latency constraints that matter in your field.**

We will reach out directly.

---

## Domain Extensions

Extending HED to a new domain is a structured process. Here is what a complete domain extension looks like.

### Step 1 — Establish the Operational Latency Constraint

Before writing any code, answer this question in writing: *In this domain, what is the maximum acceptable detection lag, and what is the cost of exceeding it?*

This answer determines the recommended λ value. It belongs in a markdown file at `docs/domains/<domain_name>.md`.

### Step 2 — Implement a Dataset Loader

Create a loader at `hed-bench/datasets/loaders/<domain_name>.py` that:
- Loads the dataset from a documented public source
- Returns a standardized tuple `(prob_stream, onset_index, labels)`
- Includes a `download.py` entry if the data requires fetching
- Documents preprocessing decisions and their justification

### Step 3 — Run the Benchmark

Use the existing benchmark runner:

```bash
python hed-bench/benchmark.py --config hed-bench/configs/<domain_name>.yaml
```

Create the config file following the pattern in `hed-bench/configs/cyber.yaml`.

### Step 4 — Document λ Calibration

Your domain extension PR must include:
- The recommended λ value
- The operational justification for that value
- A table showing HED scores at λ ∈ {0.03, 0.05, 0.10, 0.20, 0.30} on your dataset
- An update to the Hiremath Standard Table in `README.md`

### Step 5 — Submit a PR

Follow the contribution workflow below. Tag your PR with the `domain-extension` label.

---

## Development Setup

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/<your-username>/hed-score.git
cd hed-score

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install in editable mode with all development dependencies
pip install -e ".[dev,experiments]"

# Verify the installation
pytest tests/ -v

# All tests should pass before you make any changes
```

**Python version:** 3.10 or higher required.

**Dependencies:** See `pyproject.toml`. Do not add new runtime dependencies without opening an issue first. The core `hed-score` package should remain lightweight.

---

## Contribution Workflow

```
1. Open an issue describing what you intend to do.
   (Skip this for typo fixes and documentation improvements.)

2. Fork the repository.

3. Create a branch with a descriptive name:
   git checkout -b fix/streaming-baseline-recomputation
   git checkout -b feat/physiological-domain-loader
   git checkout -b docs/lambda-calibration-guide

4. Make your changes.

5. Run the full test suite:
   pytest tests/ -v

6. Run the axiom verification tests specifically:
   pytest tests/test_core.py -v -k "axiom"

7. Commit with a clear message:
   git commit -m "feat(streaming): incremental baseline correction without full recomputation"

8. Push to your fork and open a Pull Request against main.
```

### Commit Message Format

```
<type>(<scope>): <short description>

Types: feat, fix, docs, test, refactor, perf, chore
Scope: core, streaming, metrics, baseline, kernels, bench, docs

Examples:
  feat(core): add continuous HED formulation with scipy integration
  fix(baseline): correct edge case when t_star = 0
  docs(contributing): add physiological domain extension guide
  test(core): add axiom A3 decomposability verification
```

---

## Code Standards

### Type Hints

All public functions require complete type hints. Use `numpy.typing.NDArray` for array arguments.

```python
# Correct
def hed_score(
    prob_stream: NDArray[np.float64],
    t_star: int,
    lam: float = 0.10,
) -> float:

# Not acceptable
def hed_score(prob_stream, t_star, lam=0.10):
```

### Docstrings

All public functions require docstrings in the following format:

```python
def hed_score(
    prob_stream: NDArray[np.float64],
    t_star: int,
    lam: float = 0.10,
) -> float:
    """
    Compute the Hiremath Early Detection (HED) Score for a probability stream.

    Measures how early a detector places probability mass on the correct regime
    after a known change point. Implements the discrete formulation from
    arXiv:2604.04993, Section 3.

    Args:
        prob_stream:  Posterior probability of the anomalous regime at each
                      timestep. Shape: (T,). Values in [0, 1].
        t_star:       True onset index of the regime shift. Must satisfy
                      0 < t_star < len(prob_stream).
        lam:          Hiremath Decay Constant. Controls exponential penalization
                      of late detections. Default 0.10 (general purpose).
                      See README for domain-specific recommendations.

    Returns:
        hed:          Scalar HED score. Higher values indicate earlier, more
                      confident detection. Comparable across detectors evaluated
                      on the same stream with the same lam.

    Raises:
        ValueError:   If t_star is out of bounds or prob_stream contains values
                      outside [0, 1].

    References:
        Hiremath, P. S. (2026). arXiv:2604.04993 [stat.ML]
    """
```

### Formatting

```bash
# Format before committing
black hed/ tests/
isort hed/ tests/
```

Configuration lives in `pyproject.toml`. Do not modify it.

### No Silent Failures

Functions in `core.py`, `baseline.py`, and `streaming.py` must raise explicit, descriptive errors on invalid input. A function that returns `NaN` silently on a malformed probability stream is not acceptable.

```python
# Correct
if not (0 < t_star < len(prob_stream)):
    raise ValueError(
        f"t_star must satisfy 0 < t_star < len(prob_stream). "
        f"Got t_star={t_star}, len(prob_stream)={len(prob_stream)}."
    )

# Not acceptable
if t_star <= 0:
    return float("nan")
```

---

## Writing Tests

Every PR that modifies `core.py`, `baseline.py`, `kernels.py`, or `streaming.py` must include tests that verify all three axioms still hold.

### Axiom Test Templates

```python
import numpy as np
import pytest
from hed import hed_score


def test_axiom_a1_temporal_monotonicity():
    """
    A1: An earlier detection of identical quality must score higher.
    Two streams with identical post-onset peak but different rise times.
    """
    T, t_star, lam = 200, 100, 0.10

    # Fast detector: rises immediately
    p_fast = np.zeros(T)
    p_fast[t_star:t_star + 5] = np.linspace(0, 0.9, 5)
    p_fast[t_star + 5:] = 0.9

    # Slow detector: rises 20 steps later, same peak
    p_slow = np.zeros(T)
    p_slow[t_star + 20:t_star + 25] = np.linspace(0, 0.9, 5)
    p_slow[t_star + 25:] = 0.9

    assert hed_score(p_fast, t_star, lam) > hed_score(p_slow, t_star, lam), (
        "Axiom A1 violated: slower detector scored higher than faster detector."
    )


def test_axiom_a2_invariance_to_pre_attack_bias():
    """
    A2: Adding a constant offset to pre-onset probabilities must not change HED.
    """
    T, t_star, lam = 200, 100, 0.10

    p_base = np.zeros(T)
    p_base[t_star:] = 0.9

    p_biased = p_base.copy()
    p_biased[:t_star] += 0.3   # Elevated false-alarm rate before onset

    score_base = hed_score(np.clip(p_base, 0, 1), t_star, lam)
    score_biased = hed_score(np.clip(p_biased, 0, 1), t_star, lam)

    assert abs(score_base - score_biased) < 1e-10, (
        f"Axiom A2 violated: pre-onset bias changed HED score by "
        f"{abs(score_base - score_biased):.2e}."
    )


def test_axiom_a3_sensitivity_decomposability():
    """
    A3: The score must decompose into acuity, temporal lead, and calibration.
    Verified by checking that component scores multiply to the total score.
    """
    from hed.metrics import hed_decompose

    T, t_star, lam = 200, 100, 0.10
    p = np.zeros(T)
    p[t_star:] = 0.9

    total, components = hed_decompose(p, t_star, lam)
    reconstructed = components["acuity"] * components["lead"] * components["calibration"]

    assert abs(total - reconstructed) < 1e-8, (
        f"Axiom A3 violated: component product {reconstructed:.6f} != "
        f"total score {total:.6f}."
    )
```

---

## Documentation Standards

Documentation lives in `docs/`. Every public module change requires a corresponding documentation update.

For domain extensions, `docs/domains/<domain_name>.md` must include:
- A one-paragraph description of the domain and why timing matters
- The recommended λ value and its operational justification
- A worked example showing HED vs AUROC divergence on real or simulated data
- Links to the dataset loader and benchmark config

For mathematical contributions, use LaTeX notation in docstrings (`$inline$` for GitHub rendering):

```
The decay weight at step $\Delta t$ after onset is $w(\Delta t) = e^{-\lambda \Delta t}$.
```

---

## Reporting Issues

**Bug reports** should include:
- Python version and OS
- Package version (`pip show hed-score`)
- Minimal reproducible example (the shortest code that demonstrates the problem)
- Expected behavior vs actual behavior
- Whether the bug violates any of the three axioms (if you can determine this)

**Feature requests** should include:
- The use case motivating the request
- The domain you are working in
- Whether the request requires modifying the core metric or only the surrounding infrastructure

Open all issues at [github.com/prakulhiremath/hed-score/issues](https://github.com/prakulhiremath/hed-score/issues).

---

## Code of Conduct

This project operates on one principle: **the work comes first.**

Disagreements about implementation, mathematical formulation, or domain extension methodology are expected and welcome. They should be resolved by evidence, formal argument, and working code — not seniority, credential, or volume.

Contributions from students, independent researchers, and practitioners without institutional affiliation are equally valued. The HED Score was built by an undergraduate researcher because the problem was real and the existing tools were wrong. That origin is not incidental to this project. It is the point.

What is not welcome: personal attacks, dismissiveness toward contributors at any experience level, or behavior that makes the project a less welcoming place for people doing serious work.

---

## Contact

For questions about contributing, domain extensions, or the upcoming Contributors Summit:

📬 [prakulhiremath03@gmail.com](mailto:prakulhiremath03@gmail.com)
🌐 [medium.com/@prakulhiremath](https://medium.com/@prakulhiremath)
📦 [github.com/prakulhiremath/hed-score](https://github.com/prakulhiremath/hed-score)
📄 [arXiv:2604.04993](https://arxiv.org/abs/2604.04993)

---

*If your system operates in time and your evaluation framework does not, you are not measuring intelligence. You are measuring a photograph of it. Help us fix that.*
