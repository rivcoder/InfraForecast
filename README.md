# InfraForecast

**Predicting delays and cost overruns in India's central sector infrastructure projects, using official government monitoring data.**

## The Problem

India's central government tracks every infrastructure project worth ₹150 crore or more through the Ministry of Statistics and Programme Implementation (MoSPI). Their monthly Flash Reports consistently show the same story: hundreds of ongoing projects running months to years behind schedule, with cost overruns stretching into lakhs of crores. But this data lives buried in dense, multi-hundred-page PDF reports — genuinely useful information nobody outside government has time to sift through.

**InfraForecast** turns that raw PDF data into a queryable, analyzable, and predictive dataset — answering the question: *given a project's sector, state, and budget, how long will it likely take, and how much will it likely overrun?*

## What It Does

1. **Extracts** structured project-level data (sector, state, ministry, original cost, revised cost, original schedule, revised schedule) directly from official MoSPI Flash Report PDFs using `pdfplumber` — no manual data entry, no synthetic placeholders.
2. **Stores** it in a relational SQLite database, with derived fields like cost overrun %, delay in months, and expenditure ratio.
3. **Analyzes** sector-wise and state-wise overrun trends, and flags "chronic offender" projects that reappear across multiple reports with worsening numbers.
4. **Predicts** expected delay (months) and cost overrun (%) for a hypothetical new project using a Ridge regression model trained on real project characteristics.
5. **Visualizes** all of it in an interactive Streamlit dashboard, including a forecasting sandbox where you can plug in a sector/state/budget and get a prediction.

## Data Source

All data comes directly from [MoSPI's Infrastructure & Project Monitoring Division Flash Reports](https://www.mospi.gov.in/), publicly published each month, covering central sector projects costing ₹150 crore and above. This project uses a curated set of report snapshots (not all 24 months) to keep the pipeline fast without sacrificing data integrity — every number in the dashboard traces back to an actual government report, never fabricated or simulated.

## Tech Stack

- **Extraction:** Python, `pdfplumber`, `requests`
- **Storage:** SQLite via `SQLAlchemy`
- **Analysis:** `pandas`, `numpy`
- **Modeling:** `scikit-learn` (Ridge regression, one-hot encoded categorical features)
- **Visualization:** `matplotlib`, `seaborn`, `Streamlit`


## Why This Project

Most student data-analyst portfolios lean on the same handful of public Kaggle datasets. This one is built entirely from real, hard-to-parse government PDFs — the kind of messy, unglamorous data wrangling that actual analyst roles require, applied to a problem (public infrastructure accountability) that genuinely matters.

## Author

**Rashika Kaushal Jain** — Data Analyst | AI-Assisted Product Engineer
[Portfolio](https://rashikakaushaljain.qzz.io) · [GitHub](https://github.com/rivcoder)
