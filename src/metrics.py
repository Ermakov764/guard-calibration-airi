from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FlipStats:
    perturbation: str
    flip_rate: float
    n: int
    ci_low: float
    ci_high: float


def _bootstrap_flip_ci(
    flips: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile CI for mean of binary flip indicators."""
    flips = np.asarray(flips, dtype=float).ravel()
    if flips.size == 0:
        return float("nan"), float("nan")
    if flips.size == 1:
        v = float(flips[0])
        return v, v
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(flips, size=flips.size, replace=True)
        means[i] = sample.mean()
    lo = (alpha / 2) * 100
    hi = (1 - alpha / 2) * 100
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def pivot_predictions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Wide y_pred per id; clean y_true per id."""
    required = {"id", "perturbation", "y_pred", "y_true"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    wide_pred = df.pivot_table(index="id", columns="perturbation", values="y_pred", aggfunc="first")
    clean_true = df[df["perturbation"] == "clean"].set_index("id")["y_true"]
    if "clean" not in wide_pred.columns:
        raise ValueError("Scores must include perturbation 'clean'")
    return wide_pred, clean_true


def flip_rate_vs_clean(
    wide_pred: pd.DataFrame,
    perturbation: str,
    *,
    mask: pd.Series | None = None,
    n_boot: int = 2000,
) -> FlipStats:
    """Fraction of prompts where y_pred differs from clean."""
    if perturbation == "clean":
        raise ValueError("perturbation must not be 'clean'")
    if perturbation not in wide_pred.columns:
        raise ValueError(f"Missing perturbation column: {perturbation}")
    flips = (wide_pred["clean"] != wide_pred[perturbation]).astype(float)
    if mask is not None:
        flips = flips[mask.reindex(flips.index, fill_value=False)]
    flips = flips.dropna()
    rate = float(flips.mean()) if len(flips) else float("nan")
    lo, hi = _bootstrap_flip_ci(flips.values, n_boot=n_boot)
    return FlipStats(
        perturbation=perturbation,
        flip_rate=rate,
        n=int(len(flips)),
        ci_low=lo,
        ci_high=hi,
    )


def conditional_flip_rate(
    wide_pred: pd.DataFrame,
    y_true_clean: pd.Series,
    perturbation: str,
    *,
    n_boot: int = 2000,
) -> FlipStats:
    """Flip rate only where guard was correct on clean."""
    correct = wide_pred["clean"] == y_true_clean.reindex(wide_pred.index)
    return flip_rate_vs_clean(wide_pred, perturbation, mask=correct, n_boot=n_boot)


def false_positive_rate_unharmful(df: pd.DataFrame, perturbation: str) -> float:
    """P(y_pred=1 | y_true=0) for a perturbation."""
    g = df[(df["perturbation"] == perturbation) & (df["y_true"] == 0)]
    if g.empty:
        return float("nan")
    return float((g["y_pred"] == 1).mean())


def summarize_flips(df: pd.DataFrame) -> pd.DataFrame:
    """Build summary table: overall, conditional, by y_true."""
    wide_pred, y_true_clean = pivot_predictions(df)
    perts = [c for c in wide_pred.columns if c != "clean"]
    rows: list[dict] = []
    for pert in perts:
        overall = flip_rate_vs_clean(wide_pred, pert)
        cond = conditional_flip_rate(wide_pred, y_true_clean, pert)
        rows.append(
            {
                "perturbation": pert,
                "slice": "all",
                "flip_rate": overall.flip_rate,
                "ci_low": overall.ci_low,
                "ci_high": overall.ci_high,
                "n": overall.n,
            }
        )
        rows.append(
            {
                "perturbation": pert,
                "slice": "correct_on_clean",
                "flip_rate": cond.flip_rate,
                "ci_low": cond.ci_low,
                "ci_high": cond.ci_high,
                "n": cond.n,
            }
        )
        for label, name in [(0, "unharmful"), (1, "harmful")]:
            mask = y_true_clean == label
            st = flip_rate_vs_clean(wide_pred, pert, mask=mask)
            rows.append(
                {
                    "perturbation": pert,
                    "slice": name,
                    "flip_rate": st.flip_rate,
                    "ci_low": st.ci_low,
                    "ci_high": st.ci_high,
                    "n": st.n,
                }
            )
    return pd.DataFrame(rows)
