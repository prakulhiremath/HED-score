"""
models/pard_ssm.py
------------------
Minimal placeholder for PARD-SSM
(Probabilistic Anomaly and Regime Detection via Switching State-Space Models).

PARD-SSM is the *empirical vehicle* used in the HED paper to demonstrate
the metric.  It is NOT required to use HED — the metric is model-agnostic.

This placeholder defines the interface so that the package structure is
complete and users can slot in a full PARD-SSM implementation.

Full PARD-SSM couples:
  - Fractional Stochastic Differential Equations (fSDEs)
  - Switching State-Space Models (S-SSM)
  - A variational Bayes inference engine

A complete implementation is left to the full paper codebase.
See: https://arxiv.org/abs/2604.04993

NOTE
----
HED is intentionally decoupled from PARD-SSM.  Any detector that
outputs a probability stream P(t) ∈ [0, 1] can be evaluated with HED:

    from hed import hed_score
    score = hed_score(P_any_model, t_star, lam=0.1)
"""

import numpy as np
from typing import Optional


class PARDSSMDetector:
    """Minimal placeholder for PARD-SSM.

    This class defines the interface expected by the experiment scripts.
    Replace the body of fit() and predict_proba_stream() with the full
    PARD-SSM implementation from the paper.

    Parameters
    ----------
    n_regimes : int
        Number of latent regimes (typically 2: normal / anomalous).
    hurst : float
        Hurst exponent H ∈ (0, 1) for the fractional SDE component.
        H = 0.5 → standard Brownian motion.
        H > 0.5 → long-range dependence (persistent memory).
    random_state : int
    """

    def __init__(
        self,
        n_regimes: int = 2,
        hurst: float = 0.7,
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.hurst = hurst
        self.random_state = random_state
        self._fitted = False
        self._rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, **kwargs) -> "PARDSSMDetector":
        """Fit PARD-SSM to a (possibly unlabelled) time series.

        PLACEHOLDER: Stores mean and std for a trivial anomaly score.
        Replace with full fractional SDE + switching SSM inference.

        Parameters
        ----------
        X : np.ndarray, shape (T,) or (T, d)
            Observed time series.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        self._mu = np.mean(X, axis=0)
        self._sigma = np.std(X, axis=0) + 1e-8
        self._fitted = True
        print("[PARDSSMDetector] Warning: using placeholder fit(). "
              "Implement full PARD-SSM for paper-accurate results.")
        return self

    def predict_proba_stream(self, X: np.ndarray) -> np.ndarray:
        """Return P(anomalous | X_t) for each time step.

        PLACEHOLDER: Returns a z-score–based anomaly probability.
        Replace with the full posterior inference from the paper.

        Parameters
        ----------
        X : np.ndarray, shape (T,) or (T, d)

        Returns
        -------
        np.ndarray, shape (T,)
            Posterior probabilities P(anomalous) ∈ [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba_stream().")

        from hed.utils import sigmoid

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]

        z_scores = np.mean(np.abs((X - self._mu) / self._sigma), axis=1)
        # Map z-score to probability: z=0 → p≈0.27, z=3 → p≈0.95
        return sigmoid(z_scores - 1.5)

    def get_regime_posteriors(self, X: np.ndarray) -> np.ndarray:
        """Return full regime posterior matrix.

        PLACEHOLDER: Returns uniform posteriors.
        In the full model this returns the variational posterior
        q(z_t | x_{1:T}) from the switching SSM.

        Parameters
        ----------
        X : np.ndarray, shape (T,) or (T, d)

        Returns
        -------
        np.ndarray, shape (T, n_regimes)
        """
        T = len(X)
        P = np.full((T, self.n_regimes), 1.0 / self.n_regimes)
        return P
