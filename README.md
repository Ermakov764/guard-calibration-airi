# guard-calibration-airi

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ermakov764/guard-calibration-airi/blob/main/colab/AIRI_Guard_Calibration.ipynb)

Replication and extensions for [On Calibration of LLM-based Guard Models for Reliable Content Moderation](https://openreview.net/forum?id=wUbum0nd9N) (ICLR 2025): stress-test on WildGuardMix with jailbreak / paraphrase / typo, ECE on parse vs logits, flip rate and FPR.

## Colab

1. Open the notebook via the badge (GPU runtime, T4 recommended).
2. Add `HF_TOKEN` in Colab secrets if you use gated models or datasets.
3. Run all cells; download `guard_scores.csv` and `calibration_summary.json` into `outputs/`.

## Local analysis

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q

python experiments/03_plot_from_scores.py
python experiments/04_advanced_analysis.py
python experiments/05_logit_calibration.py
```

Figures are written to `report/figures/`.

## Layout

```
colab/AIRI_Guard_Calibration.ipynb
configs/experiment.yaml
src/                  # ECE, perturbations, flip/FPR
tests/
experiments/
outputs/              # scores from Colab (gitignored)
report/figures/
requirements.txt
```

## Dataset

[allenai/wildguardmix](https://huggingface.co/datasets/allenai/wildguardmix) — subsample size set in the notebook (`N_SAMPLES`).
