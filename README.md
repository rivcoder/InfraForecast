# InfraForecast
**Real-data analytics of Indian infrastructure cost & time overruns** using MoSPI Flash Reports.

[![Data Source](https://img.shields.io/badge/Data%20Source-MoSPI%20Flash%20Reports-blue)](https://mospi.gov.in)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io)

---

## What This Is
Every month, India's Ministry of Statistics (MoSPI) publishes a Flash Report listing every Central Sector infrastructure project above ₹150 crore — with original vs. revised cost, original vs. current schedule, and physical progress. This project extracts, analyzes, and models that data.

**No synthetic data.** Every number on the dashboard comes from an actual government PDF.

---

## Dashboard Pages
1. **Overview** — KPI cards (delayed projects, total COR, mean delay, worst sector) + state bar chart + delay scatter
2. **Sector Trends** — Line charts and heatmap of COR% and TOR across 5 snapshots per sector
3. **Chronic Offenders** — Projects appearing in ≥2 reports, ranked by severity score, with COR vs. TOR scatter
4. **Forecast Sandbox** — Random Forest regression: input sector + budget → predicted delay months + COR%, with validation metrics and feature importances shown alongside the prediction

---

## Model

**Features**: one-hot encoded `sector` + `log(original_cost + 1)`
**Targets**: `cost_overrun_pct` and `delay_months`
**Algorithm**: Random Forest Regressor (`n_estimators=100`, `max_depth=8`)

### Getting the validation honest

The first version of this model used Ridge regression with standard shuffled K-Fold CV, reporting R² = 0.057 (cost overrun) and R² = -0.325 (time delay) — the latter worse than just predicting the mean. Switching to Random Forest with shuffled K-Fold pushed both scores up to ~0.24, which looked like a clean win.

It wasn't. The MoSPI dataset tracks the same physical projects across multiple monthly snapshots, so shuffled splits let rows from the same project land in both train and validation folds — the model was partly recognizing projects it had already seen, not generalizing to new ones. Testing with `GroupKFold` (grouped by project identity) confirmed it: the leakage-free R² dropped to ~0.15. A further fix normalized project names before grouping (MoSPI entries have inconsistent spacing, abbreviations, and phase suffixes like "Phase I" vs "Phase-1"), tightening validation variance further.

**Final, leakage-free cross-validated scores** (`GroupKFold`, 5-fold, grouped by normalized project name):
- Cost Overrun: **R² = 0.152 (±0.022)**
- Time Delay: **R² = 0.107 (±0.044)**

Both beat the naive mean baseline on projects unseen during training. The scores are modest by design — overruns on public infrastructure are driven heavily by factors outside this dataset (land acquisition disputes, contractor solvency, regulatory delays) — so predictions are best read as **relative risk indicators**, not precise forecasts. Feature importances (Random Forest split-gain) are shown in the dashboard alongside the prediction.

---

## Data Citation
> Ministry of Statistics and Programme Implementation (MoSPI), Government of India.  
> *Flash Report on Central Sector Infrastructure Projects Costing ₹150 Crore and Above.*  
> Monthly series, April–September 2024.  
> [https://mospi.gov.in](https://mospi.gov.in)
