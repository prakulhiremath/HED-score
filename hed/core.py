"""
hed.core
========
Discrete and continuous HED Score implementations.

This is the primary computational engine.  The public entry point is
``hed_score()``, which dispatches to the discrete formulation by default.
The continuous variant is accessible via ``hed_score_continuous()``.

Discrete formulation
--------------------
::

    HED(P, t*, λ) = Σ_{t=t*}^{T-1} (P(t) - B) · exp(-λ · (t - t*))

where B = mean(P[0 : t*]) is the baseline correction term.

Continuous formulation
----------------------
::

    HED_cont(p, τ*, λ) = ∫_{τ*}^{1} (p(t) - β) · exp(-λ · (t - τ*)) dt

where β = ∫_0^{τ*} p(t) dt / τ*.

Axiom compliance
----------------
Both implementations satisfy:
- A1 (Temporal Monotonicity):  enforced by the strictly decreasing kernel.
- A2 (Pre-Attack Bias Invariance):  enforced by baseline subtraction.
- A3 (Decomposability):  exposed via ``decompose_hed()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union

import numpy as np
from numpy.typing import NDArray

from .baseline import compute_baseline, compute_continuous_baseline
from .kernels import KernelFn, exponential_kernel, get_kernel

__all__ = [
    "HEDConfig",
    "HEDResult",
    "hed_score",
    "hed_score_continuous",
    "decompose_hed",
]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class HEDConfig:
    """Configuration object for HED Score computation.

    All parameters have production-sensible defaults.  Pass a ``HEDConfig``
    to ``hed_score()`` to override individual settings without positional
    argument noise.

    Parameters
    ----------
    lam:
        Hiremath Decay Constant λ.  Controls exponential penalisation of
        late detections.  Default: 0.10 (general-purpose balanced setting).
        See the Hiremath Standard Table in the README for domain-specific
        recommendations.
    kernel:
        Kernel name (string from ``KERNEL_REGISTRY``) or a custom callable
        satisfying ``fn(delta_t, lam) -> NDArray[float64]``.
        Default: ``"exponential"``.
    normalise:
        If True, normalise the score by the theoretical maximum achievable
        given the same stream length and λ.  Produces a value closer to
        [0, 1] for easier cross-experiment comparison.  Default: False.
    clip_negative:
        If True, clamp the final score to 0.0 from below.  A negative raw
        score means the detector's post-onset output was below its own
        pre-event baseline — i.e., it actively de-prioritised the anomaly.
        Clamping loses this diagnostic signal; keep False unless you need
        a non-negative guarantee.  Default: False.
    eps:
        Small constant added to the denominator during normalisation to
        prevent division by zero.  Default: 1e-9.
    """

    lam: float = 0.10
    kernel: Union[str, KernelFn] = "exponential"
    normalise: bool = False
    clip_negative: bool = False
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if self.lam <= 0:
            raise ValueError(f"lam must be > 0, got {self.lam}")
        if isinstance(self.kernel, str):
            # Validate early — fail fast before any computation.
            get_kernel(self.kernel)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class HEDResult:
    """Container for a complete HED Score evaluation.

    Attributes
    ----------
    score:
        The primary HED scalar.
    baseline:
        The computed baseline correction B.
    t_star:
        The onset index used in the computation.
    lam:
        The decay constant used.
    n_post_onset:
        Number of timesteps in the post-onset window [t*, T).
    normalised:
        Whether ``score`` has been normalised.
    max_possible:
        Theoretical maximum score achievable with this stream and config.
        Only meaningful when ``normalised=True``; otherwise informational.
    components:
        Decomposed (Acuity, Temporal Lead, Calibration) tuple if
        ``decompose_hed()`` was called; None otherwise.
    """

    score: float
    baseline: float
    t_star: int
    lam: float
    n_post_onset: int
    normalised: bool = False
    max_possible: float = 0.0
    components: tuple[float, float, float] | None = None

    def __repr__(self) -> str:  # pragma: no cover
        norm_tag = " [normalised]" if self.normalised else ""
        return (
            f"HEDResult(score={self.score:.4f}{norm_tag}, "
            f"B={self.baseline:.4f}, t*={self.t_star}, λ={self.lam})"
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_inputs(
    prob_stream: NDArray[np.float64],
    t_star: int,
) -> None:
    """Raise informative errors for malformed inputs."""
    if prob_stream.ndim != 1:
        raise ValueError(
            f"prob_stream must be 1-D, got shape {prob_stream.shape}."
        )
    T = len(prob_stream)
    if T < 2:
        raise ValueError(
            f"prob_stream must have at least 2 timesteps, got {T}."
        )
    if not (0 <= t_star < T):
        raise ValueError(
            f"t_star={t_star} is out of bounds for stream of length {T}. "
            f"Must be in [0, {T - 1}]."
        )
    if np.any(np.isnan(prob_stream)):
        raise ValueError("prob_stream contains NaN values.")
    if np.any(np.isinf(prob_stream)):
        raise ValueError("prob_stream contains infinite values.")
    if np.any(prob_stream < 0) or np.any(prob_stream > 1):
        import warnings
        warnings.warn(
            "prob_stream contains values outside [0, 1]. "
            "HED expects calibrated probabilities.",
            UserWarning,
            stacklevel=4,
        )


# ---------------------------------------------------------------------------
# Core discrete implementation
# ---------------------------------------------------------------------------


def hed_score(
    prob_stream: NDArray[np.float64],
    t_star: int,
    lam: float = 0.10,
    *,
    kernel: Union[str, KernelFn] = "exponential",
    normalise: bool = False,
    clip_negative: bool = False,
    config: HEDConfig | None = None,
    return_result: bool = False,
) -> float | HEDResult:
    """Compute the discrete HED Score.

    Primary entry point for all HED evaluations.

    .. math::

        \\text{HED}(P,\\, t^*,\\, \\lambda) =
        \\sum_{t=t^*}^{T-1} \\bigl(P(t) - B\\bigr) \\cdot
        \\exp\\!\\bigl(-\\lambda\\,(t - t^*)\\bigr)

    Parameters
    ----------
    prob_stream:
        1-D array of detector posterior probabilities, shape (T,).
        Values should be in [0, 1].  A ``UserWarning`` is issued if they
        are not, but computation proceeds.
    t_star:
        True onset timestep index t*.  Must be in [0, T).
    lam:
        Hiremath Decay Constant λ > 0.  Ignored if ``config`` is provided.
    kernel:
        Kernel name or callable.  Ignored if ``config`` is provided.
    normalise:
        Normalise score to [0, 1] range.  Ignored if ``config`` is provided.
    clip_negative:
        Clamp score to 0 from below.  Ignored if ``config`` is provided.
    config:
        ``HEDConfig`` instance.  If provided, all other scalar parameters
        (lam, kernel, normalise, clip_negative) are ignored.
    return_result:
        If True, return a full ``HEDResult`` object instead of a scalar.

    Returns
    -------
    score or result:
        Float scalar HED score, or ``HEDResult`` if ``return_result=True``.

    Examples
    --------
    >>> import numpy as np
    >>> from hed.core import hed_score
    >>> stream = np.zeros(100)
    >>> stream[50:] = 0.9          # perfect immediate detection at t*=50
    >>> hed_score(stream, t_star=50, lam=0.10)
    8.991...
    """
    # Resolve config
    if config is not None:
        lam = config.lam
        kernel = config.kernel
        normalise = config.normalise
        clip_negative = config.clip_negative

    prob_stream = np.asarray(prob_stream, dtype=np.float64)
    _validate_inputs(prob_stream, t_star)

    # Resolve kernel callable
    kernel_fn: KernelFn = get_kernel(kernel) if isinstance(kernel, str) else kernel

    # Baseline correction (implements Axiom A2)
    B = compute_baseline(prob_stream, t_star)

    # Post-onset window
    post = prob_stream[t_star:]             # shape: (T - t*,)
    T_post = len(post)
    delta_t = np.arange(T_post, dtype=np.float64)   # [0, 1, 2, ..., T-t*-1]

    # Weights from kernel
    weights = kernel_fn(delta_t, lam)      # shape: (T_post,)

    # Weighted sum of baseline-corrected probabilities
    score = float(np.dot(post - B, weights))

    # Normalisation against theoretical maximum
    # Max score: if P(t) = 1 for all t >= t*, max = sum of weights × (1 - B)
    max_possible = float(np.sum(weights) * (1.0 - B))

    if normalise:
        eps = config.eps if config is not None else 1e-9
        score = score / (max_possible + eps)

    if clip_negative:
        score = max(score, 0.0)

    if return_result:
        return HEDResult(
            score=score,
            baseline=B,
            t_star=t_star,
            lam=lam,
            n_post_onset=T_post,
            normalised=normalise,
            max_possible=max_possible,
        )
    return score


# ---------------------------------------------------------------------------
# Continuous formulation
# ---------------------------------------------------------------------------


def hed_score_continuous(
    p_fn: NDArray[np.float64],
    tau_star: float,
    lam: float = 0.10,
    *,
    t_grid: NDArray[np.float64] | None = None,
    n_grid: int = 1000,
    kernel: Union[str, KernelFn] = "exponential",
    normalise: bool = False,
) -> float:
    """Compute the continuous HED Score via numerical integration.

    .. math::

        \\text{HED}_{\\text{cont}}(p,\\, \\tau^*,\\, \\lambda) =
        \\int_{\\tau^*}^{1}
        \\bigl(p(t) - \\beta\\bigr) \\cdot
        \\exp\\!\\bigl(-\\lambda\\,(t - \\tau^*)\\bigr)\\, dt

    Integration is performed via the trapezoidal rule on a uniform grid
    over [0, 1] unless ``t_grid`` is explicitly supplied.

    Parameters
    ----------
    p_fn:
        Sampled probability density values on the grid, shape (N,).
    tau_star:
        Continuous onset time τ* in (0, 1).
    lam:
        Hiremath Decay Constant λ > 0.
    t_grid:
        Optional explicit time grid, shape (N,), monotonically increasing
        over [0, 1].  If None, a uniform grid of ``n_grid`` points is used.
    n_grid:
        Number of grid points for the default uniform grid.  Ignored if
        ``t_grid`` is provided.
    kernel:
        Kernel name or callable.
    normalise:
        Normalise score by theoretical maximum.

    Returns
    -------
    score:
        Scalar continuous HED score.
    """
    if not (0 < tau_star < 1.0):
        raise ValueError(f"tau_star={tau_star} must be in (0, 1).")
    if lam <= 0:
        raise ValueError(f"lam must be > 0, got {lam}.")

    p_fn = np.asarray(p_fn, dtype=np.float64)
    kernel_fn: KernelFn = get_kernel(kernel) if isinstance(kernel, str) else kernel

    if t_grid is None:
        t_grid = np.linspace(0.0, 1.0, n_grid)
    else:
        t_grid = np.asarray(t_grid, dtype=np.float64)

    if len(t_grid) != len(p_fn):
        raise ValueError(
            f"t_grid length ({len(t_grid)}) must match p_fn length ({len(p_fn)})."
        )

    # Continuous baseline β
    beta = compute_continuous_baseline(p_fn, tau_star, t_grid)

    # Post-onset mask
    post_mask = t_grid >= tau_star
    t_post = t_grid[post_mask]
    p_post = p_fn[post_mask]

    if len(t_post) == 0:
        return 0.0

    delta_t = t_post - tau_star
    weights = kernel_fn(delta_t, lam)
    integrand = (p_post - beta) * weights

    score = float(np.trapz(integrand, t_post))

    if normalise:
        max_weights = kernel_fn(delta_t, lam)
        max_possible = float(np.trapz((1.0 - beta) * max_weights, t_post))
        score = score / (max_possible + 1e-9)

    return score


# ---------------------------------------------------------------------------
# Axiom A3 — decomposition
# ---------------------------------------------------------------------------


def decompose_hed(
    prob_stream: NDArray[np.float64],
    t_star: int,
    lam: float = 0.10,
    *,
    kernel: Union[str, KernelFn] = "exponential",
) -> tuple[float, float, float]:
    """Decompose HED into its three A3 components.

    Returns the (Acuity, Temporal Lead, Calibration) triple defined in
    Axiom A3 of the formal specification.

    .. math::

        \\text{HED} = \\mathcal{A} \\times \\mathcal{L} \\times \\mathcal{C}

    Definitions used here
    ~~~~~~~~~~~~~~~~~~~~~
    - **Acuity** ``A``:
        Mean post-onset probability above 0.5 (confidence of anomaly
        classification):
        ``A = mean(max(P(t) - 0.5, 0) for t in [t*, T)) / 0.5``
        Normalised to [0, 1].

    - **Temporal Lead** ``L``:
        Mass-weighted centre of detection relative to onset:
        ``L = exp(-λ · t_mass)`` where ``t_mass`` is the probability-mass-
        weighted mean timestep offset from t*.  Early detection → small
        t_mass → high L.

    - **Calibration** ``C``:
        Inverse of normalised pre-event baseline spread:
        ``C = 1 - std(P[0:t*]) / 0.5``, clamped to [0, 1].
        A well-calibrated detector (low pre-event variance) scores C ≈ 1.

    Parameters
    ----------
    prob_stream:
        1-D probability stream, shape (T,).
    t_star:
        Onset index.
    lam:
        Decay constant.
    kernel:
        Kernel name or callable.

    Returns
    -------
    (acuity, temporal_lead, calibration):
        Three floats, each in [0, 1] approximately.
    """
    prob_stream = np.asarray(prob_stream, dtype=np.float64)
    _validate_inputs(prob_stream, t_star)
    kernel_fn: KernelFn = get_kernel(kernel) if isinstance(kernel, str) else kernel

    post = prob_stream[t_star:]
    T_post = len(post)
    delta_t = np.arange(T_post, dtype=np.float64)

    # --- Acuity: mean confident detection above chance ---
    acuity_raw = np.mean(np.maximum(post - 0.5, 0.0))
    acuity = float(np.clip(acuity_raw / 0.5, 0.0, 1.0))

    # --- Temporal Lead: kernel-weighted timing ---
    weights = kernel_fn(delta_t, lam)
    total_weight = np.sum(weights)
    if total_weight > 1e-12 and np.sum(post) > 1e-12:
        # Probability-mass-weighted mean offset
        t_mass = float(np.dot(delta_t * post, weights) / (np.dot(post, weights) + 1e-12))
    else:
        t_mass = float(T_post)  # worst case
    temporal_lead = float(np.exp(-lam * t_mass))

    # --- Calibration: pre-event spread control ---
    if t_star <= 1:
        calibration = 1.0
    else:
        pre_std = float(np.std(prob_stream[:t_star], ddof=1))
        calibration = float(np.clip(1.0 - pre_std / 0.5, 0.0, 1.0))

    return acuity, temporal_lead, calibration
