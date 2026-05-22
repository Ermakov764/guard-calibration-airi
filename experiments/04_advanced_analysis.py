#!/usr/bin/env python3
"""Advanced robustness analysis and figures from guard_scores.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.metrics import (
    false_positive_rate_unharmful,
    pivot_predictions,
    summarize_flips,
)


def _plot_flip_by_slice(flip_df: pd.DataFrame, out: Path) -> None:
    plot_df = flip_df[flip_df["slice"].isin(["all", "correct_on_clean", "unharmful", "harmful"])].copy()
    slice_order = ["all", "correct_on_clean", "unharmful", "harmful"]
    plot_df["slice"] = pd.Categorical(plot_df["slice"], categories=slice_order, ordered=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=plot_df,
        x="perturbation",
        y="flip_rate",
        hue="slice",
        ax=ax,
        palette="Set2",
    )
    ax.set_ylim(0, max(0.12, plot_df["flip_rate"].max() * 1.25))
    ax.set_ylabel("Flip rate vs clean")
    ax.set_title("Prediction instability by slice")
    ax.legend(title="Slice", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "flip_rate_by_slice.png", dpi=150)
    plt.close(fig)


def _plot_flip_ci(flip_df: pd.DataFrame, out: Path) -> None:
    overall = flip_df[flip_df["slice"] == "all"].copy()
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(overall))
    ax.bar(x, overall["flip_rate"], color="coral", alpha=0.85)
    yerr_lo = overall["flip_rate"] - overall["ci_low"]
    yerr_hi = overall["ci_high"] - overall["flip_rate"]
    ax.errorbar(
        x,
        overall["flip_rate"],
        yerr=[yerr_lo, yerr_hi],
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(overall["perturbation"])
    ax.set_ylabel("Flip rate vs clean")
    ax.set_title("Flip rate with 95% bootstrap CI")
    ax.set_ylim(0, max(0.12, overall["ci_high"].max() * 1.2))
    fig.tight_layout()
    fig.savefig(out / "flip_rate_with_ci.png", dpi=150)
    plt.close(fig)


def _plot_fpr_unharmful(df: pd.DataFrame, out: Path) -> None:
    perts = ["clean", "paraphrase", "jailbreak", "typo"]
    rows = [
        {"perturbation": p, "fpr_unharmful": false_positive_rate_unharmful(df, p)}
        for p in perts
        if p in df["perturbation"].unique()
    ]
    fpr_df = pd.DataFrame(rows)
    fpr_df.to_csv(out / "fpr_unharmful.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=fpr_df, x="perturbation", y="fpr_unharmful", ax=ax, color="indianred")
    ax.set_ylim(0, 1)
    ax.set_ylabel("False positive rate (unharmful prompts)")
    ax.set_title("Over-blocking on benign prompts")
    fig.tight_layout()
    fig.savefig(out / "fpr_unharmful.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("outputs/guard_scores.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("report/figures"))
    args = parser.parse_args()

    df = pd.read_csv(args.scores)
    args.out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    flip_df = summarize_flips(df)
    flip_df.to_csv(args.out / "flip_analysis.csv", index=False)

    wide_pred, y_true_clean = pivot_predictions(df)
    summary = {
        "n_prompts": int(len(wide_pred)),
        "flip_overall": flip_df[flip_df["slice"] == "all"]
        .set_index("perturbation")["flip_rate"]
        .to_dict(),
        "flip_correct_on_clean": flip_df[flip_df["slice"] == "correct_on_clean"]
        .set_index("perturbation")["flip_rate"]
        .to_dict(),
        "flip_unharmful": flip_df[flip_df["slice"] == "unharmful"]
        .set_index("perturbation")["flip_rate"]
        .to_dict(),
        "fpr_unharmful": {
            p: false_positive_rate_unharmful(df, p)
            for p in df["perturbation"].unique()
        },
    }
    with open(args.out / "advanced_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _plot_flip_by_slice(flip_df, args.out)
    _plot_flip_ci(flip_df, args.out)
    _plot_fpr_unharmful(df, args.out)

    print(f"Wrote advanced analysis to {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
