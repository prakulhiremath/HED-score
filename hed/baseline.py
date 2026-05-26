"""
hed.baseline
============
Baseline correction term B computation for the HED Score.

The baseline B is the pre-onset mean of the detector's probability stream::

    B = (1 / t*) · Σ_{t=0}^{t*-1} P(t)

This term is subtracted from every post-onset probability before weighting,
ensuring the score reflects *probability mass gained above the detector's own
pre-event output* rather than its absolute magnitude.

Axiom A2 guarantee
------------------
For any two streams P_a and P_b that are identical over [t*, T) but differ
over [0, t*) by a constant offset δ, baseline correction guarantees::

    HED(P_a, t*, λ) = HED(P_b, t*, λ)

This module exposes both the scalar B used in the discrete formulation and the
continuous analogue β used in the continuous integral formulation.  It also
provides a rolling-window variant consumed by ``streaming.py`` for online
inference without access to the full pre-onset history.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "compute_baseline",
    "compute_continuous_baseline",
    "RollingBaseline",
]


# ---------------------------------------------------------------------------
# Discrete baseline
# ---------------------------------------------------------------------------


def compute_baseline(
    prob_stream: NDArray[np.float64],
    t_star: int,
) -> float:
    """Compute the discrete baseline correction term B.

    B is defined as the mean of the probability stream over the pre-onset
    window [0, t*).  If t* == 0 (onset at the very first timestep), there
    are no pre-onset observations; B is defined as 0.0 in this edge case.

    Parameters
    ----------
    prob_stream:
        1-D array of detector output probabilities, shape (T,).
        All values should be in [0, 1] but this is not enforced here —
        validation is the responsibility of the caller (``core.py``).
    t_star:
        True onset timestep index.  Must satisfy 0 <= t* < T.

    Returns
    -------
    B:
        Scalar baseline correction term.

    Raises
    ------
    ValueError
        If t_star is out of bounds for the given stream.

    Examples
    --------
    >>> import numpy as np
    >>> stream = np.array([0.1, 0.15, 0.12, 0.8, 0.9, 0.95])
    >>> compute_baseline(stream, t_star=3)
    0.12333333333333334
    """
    T = len(prob_stream)
    if not (0 <= t_star < T):
        raise ValueError(
            f"t_star={t_star} is out of bounds for stream of length {T}. "
            f"Must be in [0, {T - 1}]."
        )
    if t_star == 0:
        # No pre-onset window; baseline is 0 by convention.
        return 0.0
    pre_onset = prob_stream[:t_star]
    return float(np.mean(pre_onset))


def compute_baseline_std(
    prob_stream: NDArray[np.float64],
    t_star: int,
) -> float:
    """Return the standard deviation of pre-onset probabilities.

    This is an auxiliary diagnostic used by ``calibration.py`` and the
    decomposability component (Axiom A3 — Calibration term C).  Not used
    in the primary HED computation.

    Parameters
    ----------
    prob_stream:
        1-D probability stream, shape (T,).
    t_star:
        True onset timestep index.

    Returns
    -------
    std:
        Pre-onset standard deviation; 0.0 if t* <= 1.
    """
    if t_star <= 1:
        return 0.0
    return float(np.std(prob_stream[:t_star], ddof=1))


# ---------------------------------------------------------------------------
# Continuous baseline
# ---------------------------------------------------------------------------


def compute_continuous_baseline(
    p_fn: NDArray[np.float64],
    tau_star: float,
    t_grid: NDArray[np.float64],
) -> float:
    """Compute the continuous baseline correction term β.

    For a probability density p(t) sampled on a grid, β is the mean density
    over [0, τ*):

        β = ∫_0^{τ*} p(t) dt  /  τ*

    Computed via the trapezoidal rule on the provided grid.

    Parameters
    ----------
    p_fn:
        Sampled probability values, shape (N,), corresponding to ``t_grid``.
    tau_star:
        Continuous onset time in (0, 1].
    t_grid:
        Monotonically increasing time grid, shape (N,), spanning [0, 1].

    Returns
    -------
    beta:
        Scalar continuous baseline.

    Raises
    ------
    ValueError
        If tau_star is outside (0, 1].
    """
    if not (0 < tau_star <= 1.0):
        raise ValueError(
            f"tau_star={tau_star} must be in (0, 1] for the continuous formulation."
        )
    mask = t_grid < tau_star
    if not np.any(mask):
        return 0.0
    t_pre = t_grid[mask]
    p_pre = p_fn[mask]
    integral = float(np.trapz(p_pre, t_pre))
    return integral / float(tau_star)


# ---------------------------------------------------------------------------
# Rolling baseline for streaming / online inference
# ---------------------------------------------------------------------------


class RollingBaseline:
    """Online (incremental) baseline estimator for streaming HED.

    Maintains a running mean of pre-onset probabilities without storing
    the full history.  Once ``freeze()`` is called at the estimated onset,
    subsequent ``update()`` calls are ignored and ``value`` returns the
    frozen baseline.

    This is consumed by ``streaming.py``'s ``StreamingHED`` to maintain
    Axiom A2 compliance without access to the full pre-onset window.

    Parameters
    ----------
    init_value:
        Optional initial baseline estimate.  Defaults to 0.0.

    Examples
    --------
    >>> rb = RollingBaseline()
    >>> for p in [0.1, 0.15, 0.12]:
    ...     rb.update(p)
    >>> rb.freeze()
    >>> rb.value
    0.12333333333333334
    >>> rb.update(0.99)   # ignored after freeze
    >>> rb.value
    0.12333333333333334
    """

    def __init__(self, init_value: float = 0.0) -> None:
        self._sum: float = 0.0
        self._count: int = 0
        self._frozen: bool = False
        self._frozen_value: float = init_value

    def update(self, p: float) -> None:
        """Incorporate a new pre-onset probability observation.

        Parameters
        ----------
        p:
            Probability value at the current timestep.  Silently ignored
            if the baseline has already been frozen.
        """
        if self._frozen:
            return
        self._sum += float(p)
        self._count += 1

    def freeze(self) -> None:
        """Lock the baseline at its current running mean.

        Should be called at the moment the onset t* is identified (or
        estimated).  Subsequent ``update()`` calls become no-ops.
        """
        if self._frozen:
            return
        self._frozen_value = self._sum / self._count if self._count > 0 else 0.0
        self._frozen = True

    @property
    def value(self) -> float:
        """Current baseline estimate (frozen or running)."""
        if self._frozen:
            return self._frozen_value
        return self._sum / self._count if self._count > 0 else 0.0

    @property
    def is_frozen(self) -> bool:
        """True after ``freeze()`` has been called."""
        return self._frozen

    @property
    def n_observations(self) -> int:
        """Number of pre-onset samples incorporated."""
        return self._count

    def reset(self) -> None:
        """Reset to initial state (unfrozen, zero observations)."""
        self._sum = 0.0
        self._count = 0
        self._frozen = False
        self._frozen_value = 0.0
