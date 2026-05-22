"""Tests for robustness metrics."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    conditional_flip_rate,
    false_positive_rate_unharmful,
    flip_rate_vs_clean,
    pivot_predictions,
    summarize_flips,
)


def _scores_frame() -> pd.DataFrame:
    rows = []
    for pid in range(4):
        y_true = pid % 2
        clean_pred = 1 if pid < 2 else 0
        for pert, pred in [
            ("clean", clean_pred),
            ("jailbreak", clean_pred if pid != 0 else 1 - clean_pred),
        ]:
            rows.append(
                {
                    "id": pid,
                    "perturbation": pert,
                    "y_true": y_true,
                    "y_pred": pred,
                    "y_prob": 0.9,
                }
            )
    return pd.DataFrame(rows)


def test_flip_rate_vs_clean():
    df = _scores_frame()
    wide, _ = pivot_predictions(df)
    st = flip_rate_vs_clean(wide, "jailbreak", n_boot=100)
    assert st.n == 4
    assert st.flip_rate == 0.25
    assert st.ci_low <= st.flip_rate <= st.ci_high


def test_conditional_flip_only_when_correct_on_clean():
    df = _scores_frame()
    wide, y_true = pivot_predictions(df)
    st = conditional_flip_rate(wide, y_true, "jailbreak", n_boot=100)
    # ids 1,2 correct on clean; neither flips under jailbreak
    assert st.n == 2
    assert st.flip_rate == 0.0


def test_false_positive_rate_unharmful():
    df = _scores_frame()
    fpr = false_positive_rate_unharmful(df, "clean")
    # unharmful ids 0,2: preds 1,0 -> one false positive
    assert fpr == 0.5


def test_summarize_flips_columns():
    df = _scores_frame()
    summary = summarize_flips(df)
    assert set(summary["slice"]) >= {"all", "correct_on_clean", "unharmful", "harmful"}
    assert len(summary) == 4  # jailbreak only: all, correct_on_clean, unharmful, harmful
