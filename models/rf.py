"""
models/rf.py
------------
Random Forest anomaly detector.

A simple sklearn-based Random Forest wrapper that outputs posterior
probabilities for use with the HED Score.  This is a general-purpose
baseline, not the PARD-SSM model from the paper.

Usage:
    from models.rf import RFDetector
    det = RFDetector()
    det.fit(X_train, y_train)
    P = det.predict_proba_stream(X_test)
    # Then: hed_score(P, t_star, lam=0.1)
"""

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


class RFDetector:
    """Random Forest binary anomaly detector.

    Wraps sklearn's RandomForestClassifier to output a probability
    stream P(t) = P(anomalous | X_t) suitable for HED evaluation.

    Parameters
    ----------
    n_estimators : int
        Number of trees.
    max_depth : int or None
        Maximum depth of each tree.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        random_state: int = 42,
    ):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for RFDetector.  "
                "Install with: pip install scikit-learn"
            )
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFDetector":
        """Fit the detector on labelled training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,), binary labels (0/1)

        Returns
        -------
        self
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._fitted = True
        return self

    def predict_proba_stream(self, X: np.ndarray) -> np.ndarray:
        """Return P(anomalous | X_t) for each time step.

        Parameters
        ----------
        X : np.ndarray, shape (T, n_features)
            Feature matrix, one row per time step.

        Returns
        -------
        np.ndarray, shape (T,)
            Posterior probabilities in [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba_stream().")
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)
        # Column 1 = P(class=1=anomalous)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, 0]

    def feature_importances(self) -> np.ndarray:
        """Return feature importance scores from the trained forest."""
        if not self._fitted:
            raise RuntimeError("Model not fitted yet.")
        return self.model.feature_importances_
