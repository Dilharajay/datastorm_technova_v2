# Technova DataStorm V4

An end-to-end data pipeline for the Technova DataStorm competition.

The project ingests raw competition data from ZIP files, writes a bronze layer in Parquet format, and then performs a set of cleaning and enrichment steps to produce a silver layer. It also supports optional geographic validation using Sri Lanka OSM boundary data.

## What the pipeline does

1. Creates the managed data directories if they do not already exist.
2. Discovers ZIP files in `data/raw/`.
3. Extracts CSV files from each ZIP safely.
4. Writes raw tables to `data/bronze/<table_name>/data.parquet` with audit columns.
5. Reads bronze tables and applies cleaning rules.
6. Writes cleaned tables to `data/silver/<table_name>/data.parquet`.
7. Optionally downloads or uses a local Sri Lanka OSM PBF file for outlet coordinate verification.

## Repository structure

```text
technova_datastorm_v4/
├── run_pipeline.py          # Main entry point
├── scripts/                 # Training and standalone testing scripts
├── tests/                   # Unit tests for automated validation
├── src/
│   ├── configs/             # Paths and pipeline configuration
│   ├── ingest/              # Bronze-layer ingestion
│   ├── cleaning/            # Silver-layer cleaning logic
│   └── utils/               # Shared I/O and cleaning helpers
├── data/
│   ├── raw/                 # Place source ZIP files here
│   ├── bronze/              # Bronze Parquet outputs
│   ├── silver/              # Silver Parquet outputs
│   ├── gold/                # Reserved for downstream outputs
│   └── extracted/           # Temporary extraction folder
├── figures/                 # EDA / analysis figures
├── notebooks/               # Working notebooks
└── pyproject.toml
```

## Requirements

- Python 3.13+
- `pandas`
- `numpy`
- `pyarrow`
- `geopandas`
- `scipy`
- `matplotlib`

The project uses `pyproject.toml` and can be installed with your preferred Python tooling.

### Using `uv`

```powershell
uv sync
```

### Using `pip`

```powershell
python -m pip install -U pip
python -m pip install -e .
```

## Input data

Place the competition ZIP file(s) in:

```text
data/raw/
```

The ingester looks for `*.zip` files in that folder. Each ZIP should contain one or more CSV files. The expected output table name is derived from each CSV filename.

This repository already includes processed Bronze and Silver Parquet outputs under:

- `data/bronze/`
- `data/silver/`

## Running the pipeline

From the project root:

```powershell
python run_pipeline.py
```

The pipeline will:

- create any missing managed directories
- ingest raw ZIP files into bronze tables
- clean bronze tables into silver tables
- log progress to the console

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

To run the isolated unit test suite:
```powershell
pytest tests/
```

## Output locations

Each table is written using the following convention:

```text
data/<layer>/<table_name>/data.parquet
```

### Bronze layer

Bronze tables are created directly from the raw CSV files and include audit columns:

- `_layer`
- `_loaded_at`
- `_pipeline_run`

### Silver layer

Silver tables apply cleaning and normalization rules, including:

- transaction volume sanitization and flags
- outlet size/type normalization
- seasonality text cleanup
- holiday date parsing and `Year` / `Month` derivation
- outlet coordinate validation when an OSM PBF file is available

## OSM boundary file

The cleaning step for outlet coordinates can optionally use the Sri Lanka OSM extract.

- Default download URL: `https://download.geofabrik.de/asia/sri-lanka-latest.osm.pbf`
- Expected local filename: `data/raw/sri_lanka-latest.osm.pbf`

If the file is not available, the pipeline falls back to basic coordinate cleaning.

## Notebooks and analysis

The `notebooks/` folder contains exploratory and cleaning notebooks used during development. The `figures/` folder stores generated charts from EDA and validation work.

## Notes

- The pipeline is designed to be run from the repository root.
- Temporary extraction files are written to `data/extracted/` and cleaned up automatically.
- If no ZIP file is present in `data/raw/`, ingestion will stop with a clear error message.

## Extending the project

The pipeline is organized into small modules:

- `src/ingest/ingester.py` for raw ingestion
- `src/cleaning/cleaner.py` for silver-layer transformations
- `src/utils/io.py` for directory and Parquet helpers

This makes it straightforward to add a gold layer, additional cleaning rules, or export steps later.
