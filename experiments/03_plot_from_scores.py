#!/usr/bin/env python3
"""Plots from guard_scores.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.calibration import (
    expected_calibration_error,
    fit_temperature,
    reliability_bins,
    temperature_scale_probs,
)


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
    required = {"y_true", "y_prob", "perturbation"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")

    args.out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # ECE by perturbation
    rows = []
    for pert, g in df.groupby("perturbation"):
        pred = g["y_pred"].values if "y_pred" in g.columns else None
        ece = expected_calibration_error(
            g["y_true"].values, g["y_prob"].values, y_pred=pred
        )
        rows.append({"perturbation": pert, "ece": ece, "n": len(g)})
    ece_df = pd.DataFrame(rows)
    ece_df.to_csv(args.out / "ece_by_perturbation.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=ece_df, x="perturbation", y="ece", ax=ax)
    ax.set_title("ECE by perturbation type")
    ax.set_ylabel("ECE (lower is better)")
    fig.tight_layout()
    fig.savefig(args.out / "ece_by_perturbation.png", dpi=150)
    plt.close(fig)

    # Accuracy by perturbation
    acc_rows = []
    for pert, g in df.groupby("perturbation"):
        acc = (g["y_pred"] == g["y_true"]).mean() if "y_pred" in g.columns else float("nan")
        acc_rows.append({"perturbation": pert, "accuracy": acc})
    acc_df = pd.DataFrame(acc_rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=acc_df, x="perturbation", y="accuracy", ax=ax, color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy by perturbation type")
    ax.set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(args.out / "accuracy_by_perturbation.png", dpi=150)
    plt.close(fig)

    # Prediction flip rate vs clean
    if "y_pred" in df.columns:
        wide = df.pivot_table(index="id", columns="perturbation", values="y_pred")
        if "clean" in wide.columns:
            flip_rows = []
            for pert in wide.columns:
                if pert == "clean":
                    continue
                flip_rows.append(
                    {"perturbation": pert, "flip_rate": (wide["clean"] != wide[pert]).mean()}
                )
            flip_df = pd.DataFrame(flip_rows)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(data=flip_df, x="perturbation", y="flip_rate", ax=ax, color="coral")
            ax.set_title("Prediction change vs clean")
            ax.set_ylabel("Flip rate")
            ax.set_ylim(0, max(0.15, flip_df["flip_rate"].max() * 1.2))
            fig.tight_layout()
            fig.savefig(args.out / "flip_rate_vs_clean.png", dpi=150)
            plt.close(fig)

    # Reliability diagram (clean vs jailbreak)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, pert in zip(axes, ["clean", "jailbreak"]):
        g = df[df["perturbation"] == pert]
        if g.empty:
            ax.set_title(f"{pert} (no data)")
            continue
        pred = g["y_pred"].values if "y_pred" in g.columns else None
        confs, accs, _ = reliability_bins(
            g["y_true"].values, g["y_prob"].values, n_bins=10, y_pred=pred
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.plot(confs, accs, "o-", label=pert)
        ax.set_xlabel("Mean confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Reliability: {pert}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(args.out / "reliability_clean_vs_jailbreak.png", dpi=150)
    plt.close(fig)

    # Temperature scaling if logits present
    if "logit" in df.columns:
        clean = df[df["perturbation"] == "clean"]
        n_cal = max(20, int(0.3 * len(clean)))
        cal, test = clean.iloc[:n_cal], clean.iloc[n_cal:]
        t = fit_temperature(cal["logit"].values, cal["y_true"].values)
        p_test = temperature_scale_probs(test["logit"].values, t)
        pred = test["y_pred"].values if "y_pred" in test.columns else None
        ece_before = expected_calibration_error(
            test["y_true"], test["y_prob"], y_pred=pred
        )
        ece_after = expected_calibration_error(test["y_true"], p_test, y_pred=pred)
        with open(args.out / "temperature_scaling.txt", "w") as f:
            f.write(f"T={t:.3f}\nECE before={ece_before:.4f}\nECE after={ece_after:.4f}\n")

    print(f"Wrote figures to {args.out}")


if __name__ == "__main__":
    main()
