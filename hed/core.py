"""
hed/core.py
-----------
Core implementation of the Hiremath Early Detection (HED) Score.

Mathematical definition
-----------------------
Given:
  P   : array of posterior probabilities of the target regime,
        shape (T,), with values in [0, 1].
  t*  : integer index of the true regime-shift onset.
  λ   : Hiremath Decay Constant (λ_H > 0).  Controls how steeply
        late detections are penalised relative to early ones.

The HED Score is:

    HED(P, t*, λ) = Σ_{t = t*}^{T-1}  max(0, P(t) - B) · exp(-λ (t - t*))

where B = mean(P[0 : t*]) is the *baseline* — the average probability
before the shift.  Subtracting B makes the score invariant to
pre-transition bias (Axiom A2 in the paper), while the max(0, ·)
operator ensures that only positive post-shift deviations contribute.

Interpretation
--------------
- HED > 0  : detector rises above its baseline after the shift.
- HED = 0  : detector shows no post-shift improvement or remains below baseline.
- Higher λ : heavier penalty for late detection.
- Lower  λ : closer to simple post-shift integral (like AUC).

The normalised form divides by the *maximum achievable* HED for the
same λ and T, so that scores live in [0, 1] and are comparable
across different series lengths and decay constants.
"""

import numpy as np
from .utils import baseline_correct, exponential_kernel


# ---------------------------------------------------------------------------
# Primary public function
# ---------------------------------------------------------------------------

def hed_score(
    P: np.ndarray,
    t_star: int,
    lam: float = 0.1,
    normalise: bool = True,
) -> float:
    """Compute the Hiremath Early Detection (HED) Score.

    Parameters
    ----------
    P : array-like, shape (T,)
        Posterior probability stream of the target regime.
        Values must lie in [0, 1].
    t_star : int
        Index of the true regime-shift onset (0-based).
        The score is computed over t in [t_star, T).
    lam : float, default 0.1
        Hiremath Decay Constant (λ_H).  Must be > 0.
        Larger values penalise late detection more steeply.
        Recommended domain-specific defaults (Hiremath Standard Table):
          - Cyber-physical security : 0.3
          - Epidemiological monitoring : 0.05
          - Algorithmic surveillance : 0.2
          - General / exploratory : 0.1
    normalise : bool, default True
        If True, divide by the maximum achievable HED so the score
        lives in [0, 1].  A perfect detector scoring 1 means it
        output P(t) = 1 for all t >= t_star.

    Returns
    -------
    float
        The HED score.

    Examples
    --------
    >>> import numpy as np
    >>> from hed import hed_score
    >>> T, t_star = 200, 100
    >>> # A detector that jumps to 1 immediately at t_star
    >>> P_fast = np.zeros(T)
    >>> P_fast[t_star:] = 1.0
    >>> # A detector that rises slowly
    >>> P_slow = np.zeros(T)
    >>> P_slow[t_star:] = np.linspace(0, 1, T - t_star)
    >>> print(f"Fast HED: {hed_score(P_fast, t_star):.4f}")
    >>> print(f"Slow HED: {hed_score(P_slow, t_star):.4f}")
    """
    return hed_score_discrete(P, t_star, lam=lam, normalise=normalise)


# ---------------------------------------------------------------------------
# Discrete-time version (primary, vectorised)
# ---------------------------------------------------------------------------

def hed_score_discrete(
    P: np.ndarray,
    t_star: int,
    lam: float = 0.1,
    normalise: bool = True,
) -> float:
    """Discrete-time HED Score (vectorised NumPy implementation).

    This is the canonical form used in the paper.  All time steps are
    treated as unit-spaced (Δt = 1).

    Parameters
    ----------
    P : array-like, shape (T,)
    t_star : int
    lam : float
    normalise : bool

    Returns
    -------
    float
    """
    P = np.asarray(P, dtype=np.float64)
    _validate_inputs(P, t_star, lam)

    T = len(P)

    # Step 1: Baseline correction  (Axiom A2 — Invariance to Pre-Attack Bias)
    B = float(np.mean(P[:t_star])) if t_star > 0 else 0.0
    P_corrected = np.maximum(0.0, P - B)

    # tiny numerical guard (keeps everything ≥ 0)
    P_corrected = np.maximum(P_corrected, 0.0)
  
    # Step 2: Exponential decay kernel over post-shift window
    post_len = T - t_star                        # number of post-shift steps
    kernel = exponential_kernel(post_len, lam)   # exp(-λ * [0, 1, ..., T-t*-1])

    # Step 3: Weighted sum over [t_star, T)
    raw_hed = np.dot(P_corrected[t_star:], kernel)

    if not normalise:
        return float(raw_hed)

    # Normalise by the maximum achievable score:
    # Perfect detector has P_corrected(t) = 1 - B for all t >= t_star.
    # But B = 0 when pre-shift probs are 0, so max contribution per step = 1.
    # In general: max_hed = (1 - B) * sum(kernel)
    B = float(np.mean(P[:t_star])) if t_star > 0 else 0.0
    max_hed = (1.0 - B) * float(np.sum(kernel))

    if max_hed <= 0:
        return 0.0

    return float(raw_hed / max_hed)


# ---------------------------------------------------------------------------
# Continuous-time version (analytical approximation)
# ---------------------------------------------------------------------------

def hed_score_continuous(
    P: np.ndarray,
    t_star: int,
    lam: float = 0.1,
    normalise: bool = True,
) -> float:
    """Continuous-time HED Score via trapezoidal integration.

    Approximates the integral:

        HED_c = ∫_{t*}^{T} (P(t) - B) · exp(-λ(t - t*)) dt

    using the composite trapezoidal rule.  Useful when the probability
    stream is dense or when comparing against the theoretical optimum.

    Parameters
    ----------
    P : array-like, shape (T,)
    t_star : int
    lam : float
    normalise : bool

    Returns
    -------
    float
    """
    P = np.asarray(P, dtype=np.float64)
    _validate_inputs(P, t_star, lam)

    T = len(P)
    t = np.arange(T, dtype=np.float64)

    B = float(np.mean(P[:t_star])) if t_star > 0 else 0.0
    P_corrected = np.maximum(0.0, P - B)

    kernel = np.exp(-lam * (t - t_star))
    kernel[:t_star] = 0.0                        # zero out pre-shift window

    integrand = P_corrected * kernel
    raw_hed = float(np.trapz(integrand, t))

    if not normalise:
        return raw_hed

    # Max achievable: (1 - B) * ∫_{t*}^{T} exp(-λ(t-t*)) dt
    #                 = (1 - B) * (1/λ) * (1 - exp(-λ(T-t*)))
    duration = T - 1 - t_star
    max_hed = (1.0 - B) * (1.0 / lam) * (1.0 - np.exp(-lam * duration))

    if max_hed <= 0:
        return 0.0

    return float(raw_hed / max_hed)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_inputs(P: np.ndarray, t_star: int, lam: float) -> None:
    """Raise informative errors for bad inputs."""
    if P.ndim != 1:
        raise ValueError(f"P must be a 1-D array, got shape {P.shape}.")
    if not (0 <= t_star < len(P)):
        raise ValueError(
            f"t_star={t_star} is out of range for array of length {len(P)}. "
            f"Must satisfy 0 <= t_star < T."
        )
    if lam <= 0:
        raise ValueError(f"lam must be > 0, got {lam}.")
    if np.any(P < -1e-6) or np.any(P > 1 + 1e-6):
        raise ValueError(
            "P contains values outside [0, 1].  "
            "Expected posterior probabilities."
        )
    T = len(P)
    if t_star >= T - 1:
        raise ValueError(
            f"t_star={t_star} leaves no post-shift window "
            f"(T={T}).  Need at least one post-shift step."
        )
