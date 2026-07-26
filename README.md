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

## Data Pipeline

| Report | URL Fragment | Size |
|--------|-------------|------|
| April 2024 | `FlashReport_April_2024.pdf` | 5.7 MB |
| May 2024 | `FlashReport_May_2024.pdf` | 7.9 MB |
| July 2024 | `FlashReport_July_2024.pdf` | 2.3 MB |
| August 2024 | `FlashReport_August_2024.pdf` | 7.5 MB |
| September 2024 | `FlashReport_September_2024.pdf` | 8.3 MB |

**3 tables extracted per report (pdfplumber, text-based):**
- **Table 16** — "List of projects reporting additional delayed": project name, original cost, anticipated cost, original DOC, latest DOC, delay months
- **Annexure 1** — "Sector-Wise Focused Attention Projects": project name, sector, COR%, TOR months  
- **Annexure 2** — "State-Wise Summary": state, projects count, original/anticipated cost

---

## Setup

```powershell
# Full setup (downloads PDFs, builds DB, trains model, opens dashboard)
./run.ps1

# If DB already built, just open the dashboard
./run.ps1 -App
```

Or manually:

```powershell
pip install -r requirements.txt
python src/pipeline.py          # Download PDFs, parse, build DB, train models
python -m pytest src/test_pipeline.py -v   # Run tests
streamlit run src/app.py        # Open dashboard
```

---

## Project Structure

```
InfraForecast/
├── src/
│   ├── pdf_parser.py    # Downloads PDFs, extracts 3 target tables per report
│   ├── database.py      # SQLite schema (3 tables) + insert logic
│   ├── analysis.py      # Sector trends, state rankings, chronic offenders
│   ├── model.py         # Ridge regression models (COR% + delay months)
│   ├── pipeline.py      # Orchestrator: parse → insert → train
│   ├── app.py           # Streamlit 4-page dashboard
│   └── test_pipeline.py # Unit tests
├── data/
│   ├── pdfs/            # Downloaded MoSPI PDFs (cached locally)
│   ├── models/          # Saved Ridge models + feature importances
│   └── infraforecast.db # SQLite database
├── requirements.txt
└── run.ps1
```

---

## Dashboard Pages

1. **Overview** — KPI cards (delayed projects, total COR, mean delay, worst sector) + state bar chart + delay scatter
2. **Sector Trends** — Line charts and heatmap of COR% and TOR across 5 snapshots per sector
3. **Chronic Offenders** — Projects appearing in ≥2 reports, ranked by severity score, with COR vs. TOR scatter
4. **Forecast Sandbox** — Ridge regression: input sector + budget → predicted delay months + COR%

---

## Model

Ridge regression (scikit-learn) with:
- **Features**: one-hot encoded `sector` + `log(original_cost + 1)`
- **Targets**: `cost_overrun_pct` and `delay_months`
- **Cross-validated** R² reported in logs
- **Feature importance** (Ridge coefficients) displayed in the dashboard

---

## Data Citation

> Ministry of Statistics and Programme Implementation (MoSPI), Government of India.  
> *Flash Report on Central Sector Infrastructure Projects Costing ₹150 Crore and Above.*  
> Monthly series, April–September 2024.  
> [https://mospi.gov.in](https://mospi.gov.in)
