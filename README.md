# Guard Calibration — AIRI Research Proposal

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ermakov764/guard-calibration-airi/blob/main/colab/AIRI_Guard_Calibration.ipynb)

Репозиторий: [github.com/Ermakov764/guard-calibration-airi](https://github.com/Ermakov764/guard-calibration-airi)

Мини-проект под заявку **«Лето с AIRI 2026»** (трек **SafeAI**).

**Главная статья:** [On Calibration of LLM-based Guard Models (ICLR 2025)](https://openreview.net/forum?id=wUbum0nd9N)

## Colab (тяжёлый инференс)

1. Нажмите badge **Open in Colab** выше (или Colab → GitHub → `Ermakov764/guard-calibration-airi` → `colab/AIRI_Guard_Calibration.ipynb`).
2. **Runtime → Change runtime type → T4 GPU**.
3. При gated-моделях: в Colab `Secrets` добавьте `HF_TOKEN` (Hugging Face, доступ к `allenai/wildguard`).
4. Run all → скачайте `guard_scores.csv` → положите в `outputs/`.
5. Локально: `python experiments/03_plot_from_scores.py` → графики в `report/figures/`.

## Железо (ваш ASUS TUF A15)

| | |
|--|--|
| GPU | RTX 3050 Laptop **4 GB** |
| RAM | 16 GB |
| Локальная модель | **`meta-llama/Llama-Guard-3-1B`** (≈2 GB VRAM fp16) |
| Colab | **`allenai/wildguard`** 7B 4-bit |

## Структура

```
guard-calibration-proposal/
├── colab/AIRI_Guard_Calibration.ipynb   # тяжёлый инференс
├── configs/experiment.yaml
├── src/                  # ECE, perturbations, parsing
├── tests/                # unit-тесты метрик
├── experiments/
│   └── 03_plot_from_scores.py
├── report/
│   ├── PROPOSAL_OUTLINE.md
│   └── figures/          # после plot script
├── DISK_CLEANUP.md       # куда смотреть для +20 GB
└── requirements.txt
```

## Быстрый старт (локально)

```bash
cd "/home/serafim/Desktop/Статья AIRI/guard-calibration-proposal"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q
```

## После Colab

```bash
# положите guard_scores.csv в outputs/
python experiments/03_plot_from_scores.py
```

## Датасет

- [allenai/wildguardmix](https://huggingface.co/datasets/allenai/wildguardmix) — subsample 200–500 в ноутбуке.
