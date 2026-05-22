# Research Proposal — черновик (1–2 стр.)

**Трек:** SafeAI  
**Главная статья:** Wang et al., *On Calibration of LLM-based Guard Models for Reliable Content Moderation*, ICLR 2025.  
**Ссылка:** https://openreview.net/forum?id=wUbum0nd9N

---

## 1. Проблема (3–4 предложения)

LLM-based guard-модели используются для модерации prompt/response, но высокий F1 не гарантирует **калиброванную** уверенность: модель может быть **переуверенна** при ошибке. Под **jailbreak** и перефразированием калибровка ухудшается сильнее, чем ожидается по accuracy — риск для «сквозной» безопасности пайплайна.

## 2. Метод статьи (кратко)

- Оценка guard-моделей на публичных бенчмарках (prompt/response classification).
- Метрики: F1 + **ECE** (expected calibration error), reliability diagrams.
- Ключевые findings: overconfidence; jailbreak → рост miscalibration на prompt-классификации.
- Post-hoc: **temperature scaling**, **contextual calibration**.

## 3. Сильные стороны

- Прямая связь с SafeAI (модерация, атаки, надёжность confidence).
- ICLR 2025 (Core A\*), много публичных датасетов.
- Практические post-hoc методы без переобучения.

## 4. Слабые стороны

- Фокус на классификацию, не на длинную генерацию.
- Разные guard обучены на разных данных — сравнение требует осторожности.
- Post-hoc calibration может не переноситься на новые типы jailbreak.

## 5. Мини-эксперименты (наши)

**Setup:** subsample `allenai/wildguardmix` (N≈300); Colab: `allenai/wildguard` (7B, 4-bit); локальный fallback: `meta-llama/Llama-Guard-3-1B`.

| Условие | Ожидание |
|---------|----------|
| clean | baseline F1, ECE |
| paraphrase | ECE ↑ |
| jailbreak | ECE ↑↑, F1 может слегка падать |
| typo (control) | небольшие изменения |

**Развитие:** temperature scaling на logits; гибрид verbalized confidence + semantic dispersion (связь с USD/Kuhn — опционально).

## 6. Рисунки для PDF

- `report/figures/ece_by_perturbation.png`
- `report/figures/reliability_clean_vs_jailbreak.png`

## 7. Воспроизведение

```bash
cd guard-calibration-proposal
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q
# После Colab: положить outputs/guard_scores.csv
python experiments/03_plot_from_scores.py
```
