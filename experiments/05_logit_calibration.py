#!/usr/bin/env python3
"""ECE: parse vs logits vs temperature scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.calibration import expected_calibration_error, fit_temperature, temperature_scale_probs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=Path("outputs/guard_scores.csv"))
    parser.add_argument("--out", type=Path, default=Path("report/figures"))
    args = parser.parse_args()

    df = pd.read_csv(args.scores)
    if "y_prob_logit" not in df.columns or df["logit"].isna().all():
        raise SystemExit("CSV missing y_prob_logit / logit — re-run Colab notebook")

    args.out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    clean = df[df["perturbation"] == "clean"].dropna(subset=["logit"])
    n_cal = max(20, int(0.3 * len(clean)))
    cal, test = clean.iloc[:n_cal], clean.iloc[n_cal:]
    t = fit_temperature(cal["logit"].values, cal["y_true"].values)

    rows = []
    for pert in df["perturbation"].unique():
        g = df[(df["perturbation"] == pert) & (df["logit"].notna())]
        if g.empty:
            continue
        p_ts = temperature_scale_probs(g["logit"].values, t)
        pred_ts = (p_ts >= 0.5).astype(float)
        rows.append(
            {
                "perturbation": pert,
                "ece_parse": expected_calibration_error(
                    g["y_true"], g["y_prob"], y_pred=g["y_pred"]
                ),
                "ece_logits": expected_calibration_error(
                    g["y_true"], g["y_prob_logit"], y_pred=g["y_pred_logit"]
                ),
                "ece_logits_ts": expected_calibration_error(g["y_true"], p_ts, y_pred=pred_ts),
            }
        )

    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(args.out / "ece_calibration_modes.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(cmp_df))
    w = 0.25
    ax.bar([i - w for i in x], cmp_df["ece_parse"], width=w, label="parse")
    ax.bar(x, cmp_df["ece_logits"], width=w, label="logits")
    ax.bar([i + w for i in x], cmp_df["ece_logits_ts"], width=w, label="logits+TS")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cmp_df["perturbation"])
    ax.set_ylabel("ECE")
    ax.set_title(f"Calibration modes (T={t:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "ece_calibration_modes.png", dpi=150)
    plt.close(fig)

    summary = {"T_opt": t, "ece_by_pert": cmp_df.to_dict(orient="records")}
    (args.out / "logit_calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
