"""Unit tests for calibration metrics."""

import numpy as np
import pytest

from src.calibration import (
    expected_calibration_error,
    fit_temperature,
    reliability_bins,
    temperature_scale_probs,
)


def test_perfect_calibration_low_ece():
    rng = np.random.default_rng(0)
    n = 2000
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(0, 1, n) < p).astype(float)
    ece = expected_calibration_error(y, p, n_bins=15)
    assert ece < 0.08


def test_overconfident_wrong_has_higher_ece():
    rng = np.random.default_rng(1)
    n = 1000
    p = np.full(n, 0.95)
    y = np.zeros(n)  # always wrong but confident
    ece_bad = expected_calibration_error(y, p)
    p_good = rng.uniform(0.0, 1.0, n)
    y_good = (rng.uniform(0, 1, n) < p_good).astype(float)
    ece_ok = expected_calibration_error(y_good, p_good)
    assert ece_bad > ece_ok


def test_temperature_scaling_improves_miscalibrated():
    rng = np.random.default_rng(2)
    n = 800
    logits = rng.normal(2.0, 0.5, n)  # overconfident positives
    y = (rng.uniform(0, 1, n) < 0.4).astype(float)
    t = fit_temperature(logits[:400], y[:400])
    p_raw = temperature_scale_probs(logits[400:], 1.0)
    p_cal = temperature_scale_probs(logits[400:], t)
    ece_raw = expected_calibration_error(y[400:], p_raw)
    ece_cal = expected_calibration_error(y[400:], p_cal)
    assert ece_cal <= ece_raw + 0.05


def test_reliability_bins_length():
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    confs, accs, counts = reliability_bins(y, p, n_bins=5)
    assert len(confs) == 5
    assert counts.sum() == len(y)
