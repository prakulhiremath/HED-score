"""
models/lstm.py
--------------
LSTM-based anomaly detector.

A minimal PyTorch LSTM that ingests sliding windows of a time series
and outputs P(anomalous) at each step.

Requires: torch  (pip install torch)

Usage:
    from models.lstm import LSTMDetector
    det = LSTMDetector(input_size=10, hidden_size=64)
    det.fit(X_train, y_train, epochs=20)
    P = det.predict_proba_stream(X_test, window=20)
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# LSTM module
# ---------------------------------------------------------------------------

class _LSTMNet(nn.Module if _TORCH_AVAILABLE else object):
    """Internal LSTM network (binary classification head)."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.head(h_n[-1])
        return out.squeeze(-1)


# ---------------------------------------------------------------------------
# Public detector class
# ---------------------------------------------------------------------------

class LSTMDetector:
    """LSTM binary anomaly detector with sliding-window inference.

    Parameters
    ----------
    input_size : int
        Number of features per time step.
    hidden_size : int
        LSTM hidden state dimension.
    num_layers : int
        Number of stacked LSTM layers.
    window : int
        Number of time steps in each input window.
    device : str
        Torch device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        window: int = 20,
        device: str = "cpu",
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for LSTMDetector.  "
                "Install with: pip install torch"
            )
        self.window = window
        self.device = torch.device(device)
        self.model = _LSTMNet(input_size, hidden_size, num_layers).to(self.device)
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
    ) -> "LSTMDetector":
        """Train the LSTM on labelled sliding-window data.

        Parameters
        ----------
        X : np.ndarray, shape (T, input_size)
        y : np.ndarray, shape (T,), binary labels
        epochs : int
        batch_size : int
        lr : float

        Returns
        -------
        self
        """
        windows, labels = self._make_windows(X, y)
        dataset = TensorDataset(
            torch.tensor(windows, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimiser.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")

        self._fitted = True
        return self

    def predict_proba_stream(self, X: np.ndarray) -> np.ndarray:
        """Return P(anomalous | X_t) for each time step.

        Steps 0..window-1 are padded with the first window's prediction.

        Parameters
        ----------
        X : np.ndarray, shape (T, input_size)

        Returns
        -------
        np.ndarray, shape (T,)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba_stream().")

        self.model.eval()
        T = len(X)
        P = np.zeros(T)

        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            for t in range(self.window, T):
                window = X_tensor[t - self.window:t].unsqueeze(0).to(self.device)
                P[t] = self.model(window).item()

        # Fill the initial gap with the first available prediction
        if self.window < T:
            P[:self.window] = P[self.window]

        return P

    def _make_windows(self, X: np.ndarray, y: np.ndarray):
        """Slide a window over X and assign label of last step."""
        T = len(X)
        n = T - self.window
        windows = np.stack([X[i:i + self.window] for i in range(n)], axis=0)
        labels = y[self.window:]
        return windows, labels
