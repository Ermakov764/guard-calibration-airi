# GitHub + Colab

## Открыть ноутбук

- Badge в [README.md](README.md)
- Или: [Colab → Open notebook → GitHub](https://colab.research.google.com/) → `Ermakov764` / `guard-calibration-airi` / branch `main` → `colab/AIRI_Guard_Calibration.ipynb`

Прямая ссылка:

https://colab.research.google.com/github/Ermakov764/guard-calibration-airi/blob/main/colab/AIRI_Guard_Calibration.ipynb

## Hugging Face token (если модель не качается)

```python
from huggingface_hub import login
import os
login(token=os.environ["HF_TOKEN"])  # Colab Secrets
```

## После прогона

1. Скачать `guard_scores.csv`
2. В клон репозитория: `outputs/guard_scores.csv` (файл в `.gitignore`, в git не коммитить)
3. `python experiments/03_plot_from_scores.py`
