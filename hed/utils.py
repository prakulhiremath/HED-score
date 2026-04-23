"""
hed/utils.py
------------
Utility functions used by the HED core and experiment scripts.

Includes:
- baseline_correct   : subtract pre-shift mean from probability stream
- exponential_kernel : build the decay weights e^{-λ t}
- smooth_probabilities : Gaussian / exponential smoothing of raw scores
- sigmoid            : map real-valued scores to [0, 1]
"""

import numpy as np
from typing import Literal


# ---------------------------------------------------------------------------
# Baseline correction  (Axiom A2 in the paper)
# ---------------------------------------------------------------------------

def baseline_correct(P: np.ndarray, t_star: int) -> np.ndarray:
    """Subtract the pre-shift baseline from the probability stream.

    The baseline B = mean(P[0 : t_star]) represents the detector's
    average output *before* the regime shift.  Subtracting it removes
    pre-transition bias and makes HED invariant to detectors that
    have a systematically high false-alarm rate.

    If t_star == 0 there is no pre-shift window; B is set to 0.

    Parameters
    ----------
    P : np.ndarray, shape (T,)
        Raw probability stream.
    t_star : int
        Onset index of the regime shift.

    Returns
    -------
    np.ndarray, shape (T,)
        Baseline-corrected probability stream.  Values may be negative
        in the pre-shift region; that is intentional.
    """
    P = np.asarray(P, dtype=np.float64)
    B = float(np.mean(P[:t_star])) if t_star > 0 else 0.0
    return P - B


# ---------------------------------------------------------------------------
# Exponential decay kernel
# ---------------------------------------------------------------------------

def exponential_kernel(length: int, lam: float) -> np.ndarray:
    """Build an exponential decay weight vector.

    Returns w[k] = exp(-λ · k) for k = 0, 1, ..., length-1.

    Parameters
    ----------
    length : int
        Number of time steps in the post-shift window (T - t_star).
    lam : float
        Hiremath Decay Constant λ_H > 0.

    Returns
    -------
    np.ndarray, shape (length,)
        Monotonically decreasing weights starting at 1.0.

    Notes
    -----
    Implemented via np.exp on a pre-allocated integer range for
    numerical stability and vectorisation efficiency.
    """
    if length <= 0:
        return np.array([], dtype=np.float64)
    if lam <= 0:
        raise ValueError(f"lam must be > 0, got {lam}.")
    k = np.arange(length, dtype=np.float64)
    return np.exp(-lam * k)


# ---------------------------------------------------------------------------
# Probability stream smoothing
# ---------------------------------------------------------------------------

def smooth_probabilities(
    P: np.ndarray,
    method: Literal["gaussian", "exponential", "none"] = "gaussian",
    window: int = 5,
    alpha: float = 0.3,
) -> np.ndarray:
    """Smooth a raw probability / score stream.

    Useful for reducing noise in model outputs before computing HED.

    Parameters
    ----------
    P : np.ndarray, shape (T,)
        Raw probability stream.
    method : {'gaussian', 'exponential', 'none'}
        Smoothing method.
        - 'gaussian'    : uniform moving average (box filter).
        - 'exponential' : exponential moving average (EMA).
        - 'none'        : no smoothing, returns copy.
    window : int
        Half-width of the box filter (used only for 'gaussian').
    alpha : float in (0, 1]
        Smoothing factor for EMA (used only for 'exponential').
        alpha=1 → no smoothing; alpha→0 → heavy smoothing.

    Returns
    -------
    np.ndarray, shape (T,)
        Smoothed probability stream, clipped to [0, 1].
    """
    P = np.asarray(P, dtype=np.float64)

    if method == "none":
        return P.copy()

    if method == "gaussian":
        kernel = np.ones(window) / window
        smoothed = np.convolve(P, kernel, mode="same")
        # Fix boundary effects: use actual means at edges
        for i in range(window // 2):
            smoothed[i] = np.mean(P[: i + window // 2 + 1])
            smoothed[-(i + 1)] = np.mean(P[-(i + window // 2 + 1):])
        return np.clip(smoothed, 0.0, 1.0)

    if method == "exponential":
        smoothed = np.empty_like(P)
        smoothed[0] = P[0]
        for t in range(1, len(P)):
            smoothed[t] = alpha * P[t] + (1 - alpha) * smoothed[t - 1]
        return np.clip(smoothed, 0.0, 1.0)

    raise ValueError(f"Unknown smoothing method: '{method}'. "
                     f"Choose from 'gaussian', 'exponential', 'none'.")


# ---------------------------------------------------------------------------
# Sigmoid utility  (for converting raw log-odds / scores to probabilities)
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: σ(x) = 1 / (1 + exp(-x)).

    Parameters
    ----------
    x : np.ndarray
        Real-valued input.

    Returns
    -------
    np.ndarray
        Output in (0, 1).
    """
    x = np.asarray(x, dtype=np.float64)
    # Use two-branch formula to avoid overflow
    pos = x >= 0
    result = np.empty_like(x)
    result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    result[~pos] = exp_x / (1.0 + exp_x)
    return result


# ---------------------------------------------------------------------------
# Synthetic probability stream generators (used in experiments)
# ---------------------------------------------------------------------------

def make_step_detector(
    T: int,
    t_star: int,
    noise: float = 0.05,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate a fast-detection probability stream.

    Jumps to a high probability immediately at t_star, with small
    Gaussian noise throughout.  This represents an ideal early detector.

    Parameters
    ----------
    T : int
        Total time steps.
    t_star : int
        Regime-shift onset.
    noise : float
        Standard deviation of additive Gaussian noise.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray, shape (T,)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    P = np.zeros(T)
    P[t_star:] = 0.9
    P += rng.normal(0, noise, size=T)
    return np.clip(P, 0.0, 1.0)


def make_ramp_detector(
    T: int,
    t_star: int,
    noise: float = 0.05,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate a slow-detection probability stream.

    Linearly ramps from 0 to 0.9 over the post-shift window.  This
    detector has similar AUC to the step detector but substantially
    lower HED, demonstrating the metric's discriminative power.

    Parameters
    ----------
    T : int
    t_star : int
    noise : float
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray, shape (T,)
    """
    if rng is None:
        rng = np.random.default_rng(0)
    P = np.zeros(T)
    post_len = T - t_star
    P[t_star:] = np.linspace(0.0, 0.9, post_len)
    P += rng.normal(0, noise, size=T)
    return np.clip(P, 0.0, 1.0)


def make_delayed_detector(
    T: int,
    t_star: int,
    delay: int = 30,
    noise: float = 0.05,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate a delayed-detection probability stream.

    Stays near zero until t_star + delay, then jumps to 0.9.

    Parameters
    ----------
    T : int
    t_star : int
    delay : int
        Number of steps after t_star before the detector fires.
    noise : float
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray, shape (T,)
    """
    if rng is None:
        rng = np.random.default_rng(7)
    P = np.zeros(T)
    fire_at = min(t_star + delay, T - 1)
    P[fire_at:] = 0.9
    P += rng.normal(0, noise, size=T)
    return np.clip(P, 0.0, 1.0)
