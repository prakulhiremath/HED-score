"""
hed.smoothing
=============
Optional pre-smoothing operators applied to P(t) before HED scoring.

Smoothing is *not* part of the core HED definition.  It is an optional
pre-processing step for noisy probability streams where raw detector output
contains high-frequency oscillations that obscure the true detection signal.

When to smooth
--------------
- Neural detectors with unstable softmax outputs in the early post-onset window
- Streaming detectors with per-sample variance that dwarfs the regime signal
- Visualisation: smoothed curves are easier to interpret in HED timeline plots

When **not** to smooth
----------------------
- Benchmarking: apply smoothing consistently or not at all across detectors
- Axiom verification: A1/A2/A3 tests should run on unsmoothed streams to test
  the metric itself, not the pipeline

Available operators
-------------------
- ``gaussian_smooth``  — Gaussian kernel convolution (FIR, symmetric)
- ``ewma_smooth``      — Exponentially Weighted Moving Average (causal)
- ``median_smooth``    — Running median (robust to spike noise)
- ``apply_smoother``   — Unified entry point by name
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter1d

__all__ = [
    "gaussian_smooth",
    "ewma_smooth",
    "median_smooth",
    "apply_smoother",
]

SmoothMethod = Literal["gaussian", "ewma", "median"]


# ---------------------------------------------------------------------------
# Gaussian smoothing
# ---------------------------------------------------------------------------


def gaussian_smooth(
    prob_stream: NDArray[np.float64],
    sigma: float = 2.0,
    *,
    truncate: float = 4.0,
    preserve_endpoints: bool = True,
) -> NDArray[np.float64]:
    """Apply Gaussian kernel smoothing to a probability stream.

    Uses scipy's ``gaussian_filter1d`` with ``mode='reflect'`` to avoid
    boundary artefacts.  The smoothed output is clipped to [0, 1].

    Parameters
    ----------
    prob_stream:
        1-D probability array, shape (T,).
    sigma:
        Standard deviation of the Gaussian kernel in timestep units.
        Larger values → more smoothing.  Default: 2.0.
    truncate:
        Truncate the Gaussian at this many standard deviations.
        Kernel half-width = int(truncate × sigma + 0.5).
    preserve_endpoints:
        If True, keep the original values at index 0 and T-1 to avoid
        smoothing-induced distortion at the boundaries.

    Returns
    -------
    smoothed:
        Smoothed probability array, shape (T,), values in [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> stream = np.array([0.0, 0.0, 0.9, 0.1, 0.8, 0.0])
    >>> gaussian_smooth(stream, sigma=1.0)
    array([...])
    """
    from scipy.ndimage import gaussian_filter1d

    prob_stream = np.asarray(prob_stream, dtype=np.float64)
    smoothed = gaussian_filter1d(prob_stream, sigma=sigma, truncate=truncate, mode="reflect")
    smoothed = np.clip(smoothed, 0.0, 1.0)

    if preserve_endpoints and len(smoothed) >= 2:
        smoothed[0] = prob_stream[0]
        smoothed[-1] = prob_stream[-1]

    return smoothed


# ---------------------------------------------------------------------------
# EWMA smoothing
# ---------------------------------------------------------------------------


def ewma_smooth(
    prob_stream: NDArray[np.float64],
    alpha: float = 0.3,
) -> NDArray[np.float64]:
    """Apply causal Exponentially Weighted Moving Average smoothing.

    Causal means each output value depends only on current and past inputs —
    making this the only smoothing operator that is valid in true streaming
    / online settings (``streaming.py``).

    .. math::

        S_t = \\alpha \\cdot P(t) + (1 - \\alpha) \\cdot S_{t-1}

    Parameters
    ----------
    prob_stream:
        1-D probability array, shape (T,).
    alpha:
        Smoothing factor in (0, 1].  alpha=1 → no smoothing (identity).
        Smaller values → heavier smoothing with longer memory.

    Returns
    -------
    smoothed:
        EWMA-smoothed array, shape (T,), values in [0, 1].
    """
    if not (0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}.")
    prob_stream = np.asarray(prob_stream, dtype=np.float64)
    T = len(prob_stream)
    smoothed = np.empty(T, dtype=np.float64)
    smoothed[0] = prob_stream[0]
    for t in range(1, T):
        smoothed[t] = alpha * prob_stream[t] + (1.0 - alpha) * smoothed[t - 1]
    return np.clip(smoothed, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Median smoothing
# ---------------------------------------------------------------------------


def median_smooth(
    prob_stream: NDArray[np.float64],
    window: int = 5,
) -> NDArray[np.float64]:
    """Apply a running median filter to a probability stream.

    More robust to spike noise than Gaussian or EWMA smoothing because the
    median is not influenced by outlier values.  Useful for detectors that
    produce occasional probability spikes in the pre-onset period.

    Parameters
    ----------
    prob_stream:
        1-D probability array, shape (T,).
    window:
        Odd integer window size.  Even values are silently incremented to
        the next odd integer to maintain symmetry.

    Returns
    -------
    smoothed:
        Median-filtered array, shape (T,), values in [0, 1].
    """
    from scipy.signal import medfilt

    prob_stream = np.asarray(prob_stream, dtype=np.float64)
    if window % 2 == 0:
        window += 1  # medfilt requires odd kernel
    smoothed = medfilt(prob_stream, kernel_size=window)
    return np.clip(smoothed, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def apply_smoother(
    prob_stream: NDArray[np.float64],
    method: SmoothMethod = "gaussian",
    **kwargs,
) -> NDArray[np.float64]:
    """Apply a named smoothing operator to a probability stream.

    Parameters
    ----------
    prob_stream:
        1-D probability array, shape (T,).
    method:
        One of ``"gaussian"``, ``"ewma"``, or ``"median"``.
    **kwargs:
        Forwarded to the underlying smoother function.
        - gaussian: sigma, truncate, preserve_endpoints
        - ewma: alpha
        - median: window

    Returns
    -------
    smoothed:
        Smoothed probability array, shape (T,).

    Raises
    ------
    ValueError
        If *method* is not one of the supported options.

    Examples
    --------
    >>> import numpy as np
    >>> stream = np.random.default_rng(0).uniform(0, 0.2, 200)
    >>> stream[100:] += 0.7
    >>> stream = np.clip(stream, 0, 1)
    >>> smoothed = apply_smoother(stream, method="ewma", alpha=0.2)
    """
    _dispatch = {
        "gaussian": gaussian_smooth,
        "ewma": ewma_smooth,
        "median": median_smooth,
    }
    if method not in _dispatch:
        raise ValueError(
            f"Unknown smoothing method {method!r}. "
            f"Choose from: {', '.join(sorted(_dispatch))}."
        )
    return _dispatch[method](prob_stream, **kwargs)
