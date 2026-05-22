"""Expected Calibration Error and reliability diagrams (ICLR guard paper style)."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    y_pred: np.ndarray | None = None,
) -> float:
    """
    ECE for binary classification (classifier calibration).
    y_true: 0/1 ground truth
    y_prob: predicted P(y=1), in [0, 1]
    y_pred: optional hard predictions; default threshold 0.5 on y_prob
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(float)
    else:
        y_pred = np.asarray(y_pred, dtype=float).ravel()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        acc = (y_pred[mask] == y_true[mask]).mean()
        # Confidence in the predicted class (standard binary classifier ECE).
        conf = np.where(
            y_pred[mask] == 1,
            y_prob[mask],
            1.0 - y_prob[mask],
        ).mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    y_pred: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean_confidence, accuracy, count) per bin."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(float)
    else:
        y_pred = np.asarray(y_pred, dtype=float).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    confs, accs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            confs.append(np.nan)
            accs.append(np.nan)
            counts.append(0)
            continue
        confs.append(
            float(
                np.where(
                    y_pred[mask] == 1,
                    y_prob[mask],
                    1.0 - y_prob[mask],
                ).mean()
            )
        )
        accs.append(float((y_pred[mask] == y_true[mask]).mean()))
        counts.append(int(mask.sum()))
    return np.array(confs), np.array(accs), np.array(counts)


def temperature_scale_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Binary logits -> calibrated probability via temperature scaling."""
    logits = np.asarray(logits, dtype=float).ravel()
    t = max(temperature, 1e-6)
    return 1.0 / (1.0 + np.exp(-logits / t))


def fit_temperature(
    logits: np.ndarray,
    y_true: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    """Grid search T minimizing ECE on calibration split."""
    if grid is None:
        grid = np.linspace(0.05, 5.0, 100)
    best_t, best_ece = 1.0, float("inf")
    for t in grid:
        p = temperature_scale_probs(logits, t)
        ece = expected_calibration_error(y_true, p)
        if ece < best_ece:
            best_ece = ece
            best_t = float(t)
    return best_t
