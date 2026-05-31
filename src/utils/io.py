# Shared I/O helpers

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.configs.config import config

log = logging.getLogger("pipeline.utils")


# Directory management
def ensure_dirs(*dirs: Path) -> None:
    """
    Ensure that the given directories exist, creating them if necessary, and log the action.
    """

    for d in dirs:
        existed = d.exists()
        d.mkdir(parents=True, exist_ok=True)
        if existed:
            log.info("Directory already exists: %s", d)
        else:
            log.info("Created directory: %s", d)


# Audit columns
def add_audit_columns(df: pd.DataFrame, layer: str) -> pd.DataFrame:
    """
    Attach _layer, _loaded_at, _pipeline_run to any DataFrame.
    returns a new DataFrame with the additional columns, without modifying the original.
    """
    now = datetime.now(timezone.utc)
    df = df.copy()
    df[config.col_layer] = layer
    df[config.col_loaded_at] = now.isoformat()
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
    Stamps the correct ``_layer`` audit column before writing.
    Returns the number of rows written.
    """
    data = add_audit_columns(df, layer)
    out_path = base_dir / table_name
    out_path.mkdir(parents=True, exist_ok=True)
    parquet_path = out_path / "data.parquet"
    data.to_parquet(
        parquet_path,
        index=False,
        compression=config.parquet_compression,
        engine="pyarrow",
    )
    log.info(
        "[%s] Wrote parquet: table='%s', rows=%d, path=%s",
        layer.upper(),
        table_name,
        len(data),
        parquet_path,
    )
    return len(data)


def write_rejects(
    df: pd.DataFrame,
    base_dir: Path,
    table_name: str,
    layer: str,
    reject_reason: Optional[str] = None,
) -> int:
    """
    Write rejected records to the rejects layer.

    If the DataFrame already has a ``_reject_reason`` column it is
    preserved; otherwise ``reject_reason`` is used as a uniform label.
    Fresh audit columns are stamped regardless.
    """
    data = df.copy()
    if "_reject_reason" not in data.columns:
        data["_reject_reason"] = (
            reject_reason if reject_reason is not None else "unknown"
        )
    data = add_audit_columns(data, layer)
    out_path = base_dir / table_name
    out_path.mkdir(parents=True, exist_ok=True)
    parquet_path = out_path / "data.parquet"
    data.to_parquet(
        parquet_path,
        index=False,
        compression=config.parquet_compression,
        engine="pyarrow",
    )
    label = data["_reject_reason"].iloc[0] if len(data) else "none"
    log.warning(
        "[%s] Wrote %d rejected row(s) to %s (reason: %s)",
        layer.upper(),
        len(data),
        parquet_path,
        label,
    )
    return len(data)


def read_parquet(base_dir: Path, table_name: str) -> pd.DataFrame:
    """
    Read a Parquet file from the standard location and log the action.
    Returns the loaded DataFrame.
    """
    parquet_path = base_dir / table_name / "data.parquet"
    log.info("Reading parquet: table='%s', path=%s", table_name, parquet_path)
    return pd.read_parquet(parquet_path)
