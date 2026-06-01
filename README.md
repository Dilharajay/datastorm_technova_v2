# Technova DataStorm V4

An end-to-end data pipeline for the Technova DataStorm v7.0 Final Round competition.

The project implements a Medallion Lakehouse architecture (Bronze → Silver → Gold) to estimate latent (unconstrained) demand for 20,000 traditional trade outlets in Sri Lanka, allocate a LKR 5M promotional budget, and provide an interactive intelligence dashboard.

## Dashboard Preview

![Dashboard overview](assets/1.avif)
![Dashboard overview](assets/2.avif)

## What the pipeline does

1. **Bronze** — Ingests raw competition ZIP files, extracts CSVs, writes Parquet with audit columns.
2. **Silver** — Cleans and normalizes 6 tables: transactions, outlet master, outlet coordinates, distributor seasonality, holidays, and POI scores. Also computes **competitive catchment density** (outlets within 500m).
3. **Gold** — Builds a fact table merging all silver tables for modeling.
4. **Budget Optimization** — PuLP linear program allocating LKR 5M across Western Province outlets to maximize incremental volume.
5. **XAI** — Generates human-readable explanations for each outlet's predicted score (Ollama local LLM with template fallback).

## Repository structure

```text
technova_datastorm_v4/
├── run_pipeline.py              # Main entry point
├── app.py                       # Streamlit web app
├── src/
│   ├── configs/                 # Paths and pipeline configuration
│   ├── ingest/                  # Bronze-layer ingestion
│   ├── cleaning/                # Silver-layer cleaning logic
│   ├── optimization/            # Budget optimizer (PuLP)
│   ├── xai/                     # Outlet explainer (Ollama)
│   └── utils/                   # I/O, cleaning, eda, POI, catchment helpers
├── data/
│   ├── raw/                     # Place source ZIP files here
│   ├── bronze/                  # Bronze Parquet outputs
│   ├── silver/                  # Silver Parquet outputs
│   ├── gold/                    # Gold fact table + predictions
│   └── rejects/                 # Rejected records store
├── figures/                     # EDA / analysis figures
├── notebooks/                   # EDA and modeling notebooks
│   ├── eda.ipynb                # Exploratory data analysis
│   ├── model_pytorch_faster.ipynb  # Latent demand estimation
│   └── check_gold.ipynb         # Gold layer validation
└── pyproject.toml
```

## Requirements

- Python 3.13+
- Dependencies managed via `uv` (see `pyproject.toml`)

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

> Note: `torch` is excluded from default deps. For the Tobit baseline in the model notebook, install manually:
> ```bash
> uv add torch
> ```

### 2. Run the full pipeline

```bash
uv run python run_pipeline.py
```

This runs: Bronze ingester → Silver cleaner (incl. catchment density) → Gold fact table → Budget optimizer → XAI explanations.

### 3. Run notebooks

```bash
# EDA (includes competition density analysis)
uv run jupyter nbconvert --to notebook --execute notebooks/eda.ipynb --output eda_executed.ipynb

# Model (latent demand estimation)
uv run jupyter nbconvert --to notebook --execute notebooks/model_pytorch_faster.ipynb --output model_executed.ipynb

# Gold validation
uv run jupyter nbconvert --to notebook --execute notebooks/check_gold.ipynb --output check_gold_executed.ipynb
```

### 4. Launch the web app

```bash
uv run streamlit run app.py
```

## Deliverables

| File | Description |
|------|-------------|
| `data/teamname_predictions.csv` | Outlet_ID + Maximum_Monthly_Liters for Jan 2026 |
| `reports/teamname_budget_allocations.csv` | Western Province trade spend allocation |
| `reports/teamname_outlet_explanations.csv` | Per-outlet XAI narratives |
| `notebooks/predictions_jan2026.parquet` | Full prediction output with confidence intervals |

## Key features

- **Competitive Catchment Density** — counts competing outlets within 500m using spatial joins (EPSG:5235 metric CRS)
- **POI Distance-Decay Scoring** — exponential decay scores for schools, hospitals, bus stops, tourist attractions
- **Censoring Detection** — 6 proxy rules detecting supply-constrained observations
- **Latent Demand Model** — Two-stage: Tobit + XGBoost on de-censored series (MAE 4.87, R² 0.985)
- **Budget Optimization** — PuLP LP solver maximizing incremental volume under LKR 5M constraint
- **XAI** — Ollama-generated business explanations per outlet
- **Streamlit Dashboard** — Browse predictions, filter by province/distributor, drill into outlet detail

By default, this will run both the data extraction/cleaning (ETL) and the model training. You can selectively run parts of the pipeline using command-line arguments:

```powershell
# Skip the ETL stages (Bronze, Silver, Gold) and only run model training
python run_pipeline.py --skip-etl

# Skip model training and only run the ETL stages
python run_pipeline.py --skip-model
```

## Testing and Validation

To quickly verify dependencies and the core functionality of the Latent Demand Model (without running the full data ingestion pipeline), you can use the provided demo script:

```powershell
python scripts/test_latent_demand_model.py
```

## Output locations

Each table is written using the convention:

```text
data/<layer>/<table_name>/data.parquet
```

Managed layers:
- `data/bronze/` — raw ingested tables
- `data/silver/` — cleaned tables with audit columns
- `data/gold/` — fact table, predictions, budget allocations
- `data/rejects/` — rejected records with `_reject_reason`
