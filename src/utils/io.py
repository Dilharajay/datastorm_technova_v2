# Shared I/O helpers

import logging
from datetime import datetime, timezone
from pathlib import Path
 
import pandas as pd
 
from src.configs.config import config
 
log = logging.getLogger("pipeline.utils")

# Directory management
def ensure_dirs(*dirs: Path) -> None:
    """
    Ensure that the given directories exist, creating them if necessary, and log the action.
    """

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        log.info("Directory ready: %s", d)
 
 
# Audit columns 
def add_audit_columns(df: pd.DataFrame, layer: str) -> pd.DataFrame:
    """
    Attach _layer, _loaded_at, _pipeline_run to any DataFrame.
    returns a new DataFrame with the additional columns, without modifying the original.
    """
    now = datetime.now(timezone.utc)
    df = df.copy()
    df[config.col_layer]        = layer
    df[config.col_loaded_at]    = now.isoformat()
    df[config.col_pipeline_run] = now.strftime("%Y%m%d_%H%M%S")
    return df
 
 
def drop_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop audit columns if they exist, to get back to the original source schema. Useful for testing and re-ingestion.
    Returns a new DataFrame without modifying the original.
    """

    cols = [config.col_layer, config.col_loaded_at, config.col_pipeline_run]
    return df.drop(columns=[c for c in cols if c in df.columns])
 
 
# Parquet I/O
def write_parquet(df: pd.DataFrame, base_dir: Path, table_name: str, layer: str) -> int:
    """
    Write a DataFrame to Parquet with a standard filename and compression, and log the action.
    Returns the number of rows written.
    """
    out_path = base_dir / table_name
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path / "data.parquet",index=False,compression=config.parquet_compression,engine="pyarrow",)
    log.info("[%s] %-24s → %s  (%d rows)", layer.upper(), table_name, out_path, len(df))
    return len(df)
 
 
def read_parquet(base_dir: Path, table_name: str) -> pd.DataFrame:
    """
    Read a Parquet file from the standard location and log the action.
    Returns the loaded DataFrame.
    """

    return pd.read_parquet(base_dir / table_name / "data.parquet", engine="pyarrow")