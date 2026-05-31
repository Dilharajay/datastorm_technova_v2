"""
Reusable cleaning helpers for the Silver layer.

These functions are intentionally composable:
- low-level transforms clean individual columns
- dataset-level wrappers apply the notebook logic end-to-end
"""

from __future__ import annotations

import logging
from pathlib import Path
import tempfile
from typing import Optional
import urllib.request

import numpy as np
import pandas as pd

import geopandas as gpd
from shapely.geometry import Point

log = logging.getLogger("pipeline.cleaning")


def _log_missing_column(column: str, action: str) -> None:
    """Log a consistent warning when a required column is missing."""
    log.warning("Skipped %s because column '%s' was not found", action, column)


def _log_row_change(action: str, before_rows: int, after_rows: int, details: str = "") -> None:
    """Log a compact summary for row-filtering operations."""
    removed = before_rows - after_rows
    suffix = f" {details}" if details else ""
    log.info("%s: kept %d/%d rows (%d removed)%s", action, after_rows, before_rows, removed, suffix)


# Low level helpers
def normalize_text_column(
    df: pd.DataFrame,
    column: str,
    *,
    strip: bool = True,
    lower: bool = True,
    collapse_whitespace: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Return a DataFrame with one text column normalized.
    """
    out = df.copy() if copy else df
    if column not in out.columns:
        _log_missing_column(column, "text normalization")
        return out

    s = out[column].astype("string")
    if collapse_whitespace:
        s = s.str.replace(r"\s+", " ", regex=True)
    if strip:
        s = s.str.strip()
    if lower:
        s = s.str.lower()

    out[column] = s.astype("object")
    log.info(
        "Normalized text column '%s' (collapse_whitespace=%s, strip=%s, lower=%s)",
        column,
        collapse_whitespace,
        strip,
        lower,
    )
    return out


def make_abs_columns(df: pd.DataFrame, columns: list[str], copy: bool = True) -> pd.DataFrame:
    """
    Convert selected numeric columns to absolute values.
    """
    out = df.copy() if copy else df
    converted: list[str] = []
    missing: list[str] = []

    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].abs()
            converted.append(col)
        else:
            missing.append(col)

    if converted:
        log.info("Converted numeric columns to absolute values: %s", ", ".join(converted))
    if missing:
        log.warning("Skipped absolute-value conversion for missing columns: %s", ", ".join(missing))
    return out


def filter_numeric_upper_bound(
    df: pd.DataFrame,
    column: str,
    upper_bound: float,
    *,
    copy: bool = True,
    inclusive: bool = False,
) -> pd.DataFrame:
    """Return rows whose numeric value is below an upper bound."""
    out = df.copy() if copy else df
    if column not in out.columns:
        _log_missing_column(column, "upper-bound filtering")
        return out

    before_rows = len(out)
    values = pd.to_numeric(out[column], errors="coerce")
    mask = values <= upper_bound if inclusive else values < upper_bound
    filtered = out.loc[mask].copy()

    _log_row_change(
        f"Applied upper-bound filter on '{column}'",
        before_rows,
        len(filtered),
        f"(bound={upper_bound}, inclusive={inclusive})",
    )
    return filtered


def filter_numeric_range(
    df: pd.DataFrame,
    column: str,
    lower_bound: float,
    upper_bound: float,
    *,
    copy: bool = True,
    inclusive: bool = True,
) -> pd.DataFrame:
    """Return rows within a numeric range."""
    out = df.copy() if copy else df
    if column not in out.columns:
        _log_missing_column(column, "range filtering")
        return out

    before_rows = len(out)
    values = pd.to_numeric(out[column], errors="coerce")
    if inclusive:
        mask = (values >= lower_bound) & (values <= upper_bound)
    else:
        mask = (values > lower_bound) & (values < upper_bound)

    filtered = out.loc[mask].copy()
    _log_row_change(
        f"Applied range filter on '{column}'",
        before_rows,
        len(filtered),
        f"(bounds=({lower_bound}, {upper_bound}), inclusive={inclusive})",
    )
    return filtered


def parse_date_column(
    df: pd.DataFrame,
    column: str,
    *,
    copy: bool = True,
    errors: str = "coerce",
) -> pd.DataFrame:
    """Parse a date column in-place and return the DataFrame."""
    out = df.copy() if copy else df
    if column in out.columns:
        out[column] = pd.Series(pd.to_datetime(out[column].to_numpy(), errors=errors), index=out.index)
        log.info("Parsed date column '%s' (errors=%s)", column, errors)
    else:
        _log_missing_column(column, "date parsing")
    return out


def add_year_month_from_date(
    df: pd.DataFrame,
    date_column: str = "Date",
    year_column: str = "Year",
    month_column: str = "Month",
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Create year/month helper columns from a datetime column."""
    out = df.copy() if copy else df
    if date_column not in out.columns:
        _log_missing_column(date_column, "year/month derivation")
        return out

    date_values = pd.Series(pd.to_datetime(out[date_column].to_numpy(), errors="coerce"), index=out.index)
    out[year_column] = pd.Series(
        [value.year if not pd.isna(value) else pd.NA for value in date_values],
        index=out.index,
        dtype="Int64",
    )
    out[month_column] = pd.Series(
        [value.month if not pd.isna(value) else pd.NA for value in date_values],
        index=out.index,
        dtype="Int64",
    )
    log.info("Derived calendar columns '%s' and '%s' from '%s'", year_column, month_column, date_column)
    return out


def remove_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    whisker: float = 1.5,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Remove outliers from a numeric column using the IQR rule.

    Returns a filtered DataFrame, not just a Series.
    """
    if column not in df.columns:
        _log_missing_column(column, "IQR outlier removal")
        return df.copy()

    data = df.copy()
    series = pd.Series(pd.to_numeric(data[column], errors="coerce"), index=data.index)
    before_rows = len(data)

    if dropna:
        non_numeric_rows = int(series.isna().sum())
        data = data.loc[series.notna()].copy()
        series = pd.Series(pd.to_numeric(data[column], errors="coerce"), index=data.index)
        if non_numeric_rows:
            log.info("Excluded %d non-numeric row(s) from '%s' before IQR calculation", non_numeric_rows, column)

    if series.empty:
        log.warning("Skipped IQR outlier removal for '%s' because no numeric data remained", column)
        return data

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        log.info("Skipped IQR outlier removal for '%s' because the IQR is 0 or NaN", column)
        return data

    lower_bound = q1 - whisker * iqr
    upper_bound = q3 + whisker * iqr

    mask = series.between(lower_bound, upper_bound, inclusive="both")
    if not dropna:
        mask = mask | series.isna()

    filtered = data.loc[mask].copy()
    _log_row_change(
        f"Applied IQR outlier filter on '{column}'",
        before_rows,
        len(filtered),
        f"(whisker={whisker}, dropna={dropna})",
    )
    return filtered


def flag_zero_volume_rows(df: pd.DataFrame, volume_col: str = "Volume_Liters") -> pd.Series:
    """Flag rows where Volume_Liters == 0."""
    return pd.Series(pd.to_numeric(df[volume_col], errors="coerce"), index=df.index) == 0


def flag_flat_volume_outlets(
    df: pd.DataFrame,
    outlet_col: str = "Outlet_ID",
    volume_col: str = "Volume_Liters",
    min_months: int = 3,
    tolerance: float = 0.01,
) -> pd.Series:
    """
    Flag outlets whose Volume_Liters is suspiciously constant across months.
    Returns a boolean Series aligned to df's index.
    """
    work = df[[outlet_col, volume_col]].copy()
    work[volume_col] = pd.Series(pd.to_numeric(work[volume_col], errors="coerce"), index=work.index)

    outlet_stats = work.groupby(outlet_col)[volume_col].agg(["std", "mean", "count"]).reset_index()
    outlet_stats["cv"] = outlet_stats["std"] / outlet_stats["mean"].replace(0, np.nan)
    outlet_stats["cv"] = outlet_stats["cv"].fillna(0)

    flat_outlets = outlet_stats.loc[
        (outlet_stats["count"] >= min_months) & (outlet_stats["cv"] < tolerance),
        outlet_col,
    ].tolist()

    if flat_outlets:
        log.info(
            "Flagged %d outlet(s) as flat-volume patterns (min_months=%d, tolerance=%s)",
            len(flat_outlets),
            min_months,
            tolerance,
        )
    else:
        log.info(
            "No flat-volume outlets detected (min_months=%d, tolerance=%s)",
            min_months,
            tolerance,
        )

    return df[outlet_col].isin(flat_outlets)


def fix_coords(
    row: pd.Series,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    *,
    lat_min: float = 5.9,
    lat_max: float = 9.9,
    lon_min: float = 79.5,
    lon_max: float = 81.9,
) -> pd.Series:
    """
    Fix swapped Sri Lanka coordinates and label rows that are valid / invalid.

    Returns a Series with:
    - Latitude
    - Longitude
    - coord_status
    """
    lat = row[lat_col]
    lon = row[lon_col]

    if (lon_min <= lat <= lon_max) and (lat_min <= lon <= lat_max):
        return pd.Series({lat_col: lon, lon_col: lat, "coord_status": "Swapped and Fixed"})

    if (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max):
        return pd.Series({lat_col: lat, lon_col: lon, "coord_status": "Valid"})

    return pd.Series({lat_col: lat, lon_col: lon, "coord_status": "Out of Bounds"})


def clean_outlet_coordinates_basic(
    df: pd.DataFrame,
    *,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    status_col: str = "coord_status",
    valid_statuses: tuple[str, ...] = ("Valid", "Swapped and Fixed"),
) -> pd.DataFrame:
    """Apply notebook coordinate cleaning (drop nulls, numeric cast, swap-fix, valid filter)."""
    clean = df.copy()
    before_rows = len(clean)

    if lat_col not in clean.columns or lon_col not in clean.columns:
        missing = [col for col in (lat_col, lon_col) if col not in clean.columns]
        log.warning("Skipped outlet coordinate cleaning; missing column(s): %s", ", ".join(missing))
        return clean

    clean = clean.dropna(subset=[lat_col, lon_col]).copy()
    clean[lat_col] = pd.to_numeric(clean[lat_col], errors="coerce")
    clean[lon_col] = pd.to_numeric(clean[lon_col], errors="coerce")
    clean = clean.dropna(subset=[lat_col, lon_col]).copy()

    geo_status = clean.apply(fix_coords, axis=1, lat_col=lat_col, lon_col=lon_col)
    clean[lat_col] = geo_status[lat_col]
    clean[lon_col] = geo_status[lon_col]
    clean[status_col] = geo_status["coord_status"]

    clean = clean[clean[status_col].isin(valid_statuses)].copy()
    _log_row_change(
        "Cleaned outlet coordinates",
        before_rows,
        len(clean),
        f"(valid_statuses={valid_statuses})",
    )
    return clean


def outlet_coordinates_to_geodataframe(
    df: pd.DataFrame,
    *,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Convert cleaned outlet coordinates to a GeoDataFrame."""
    if lat_col not in df.columns or lon_col not in df.columns:
        missing = [col for col in (lat_col, lon_col) if col not in df.columns]
        raise ValueError(f"Cannot create GeoDataFrame; missing column(s): {', '.join(missing)}")

    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=crs)
    log.info("Converted outlet coordinates to GeoDataFrame with %d row(s)", len(gdf))
    return gdf


def load_administrative_boundaries_from_pbf(
    pbf_path: str | Path,
    *,
    layer: str = "multipolygons",
    boundary_col: str = "boundary",
    boundary_value: str = "administrative",
    keep_columns: tuple[str, ...] = ("name", "admin_level", "geometry"),
) -> gpd.GeoDataFrame:
    """Load administrative boundaries from an OSM PBF file."""
    pbf_path = Path(pbf_path)
    if not pbf_path.exists():
        raise FileNotFoundError(f"PBF file not found: {pbf_path}")

    polygons = gpd.read_file(pbf_path, layer=layer)
    boundaries = polygons[polygons[boundary_col] == boundary_value].copy()

    available = [col for col in keep_columns if col in boundaries.columns]
    if "geometry" not in available:
        available.append("geometry")
    boundaries = boundaries[available].copy()

    log.info("Loaded %d administrative boundary polygon(s) from %s", len(boundaries), pbf_path)
    return boundaries


def spatially_verify_outlet_points(
    points_gdf: gpd.GeoDataFrame,
    boundaries_gdf: gpd.GeoDataFrame,
    *,
    how: str = "inner",
    predicate: str = "within",
) -> gpd.GeoDataFrame:
    """Spatial join to keep only points inside administrative boundaries."""
    boundaries = boundaries_gdf
    if points_gdf.crs != boundaries.crs:
        boundaries = boundaries.to_crs(points_gdf.crs)
        log.info("Aligned boundary CRS from %s to %s", boundaries_gdf.crs, points_gdf.crs)

    verified = gpd.sjoin(points_gdf, boundaries, how=how, predicate=predicate)
    log.info(
        "Spatially verified outlet points: kept %d/%d row(s)",
        len(verified),
        len(points_gdf),
    )
    return verified


def clean_and_verify_outlet_coordinates(
    df: pd.DataFrame,
    pbf_path: str | Path,
    *,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    drop_geometry: bool = True,
) -> pd.DataFrame:
    """End-to-end outlet geo cleaning from notebook: clean coords + PBF boundary verification."""
    clean = clean_outlet_coordinates_basic(df, lat_col=lat_col, lon_col=lon_col)
    points = outlet_coordinates_to_geodataframe(clean, lat_col=lat_col, lon_col=lon_col)
    boundaries = load_administrative_boundaries_from_pbf(pbf_path)
    verified = spatially_verify_outlet_points(points, boundaries)

    if drop_geometry and "geometry" in verified.columns:
        verified = pd.DataFrame(verified.drop(columns=["geometry"]))
    else:
        verified = pd.DataFrame(verified)

    return verified


def download_file(url: str, dest: Path, *, chunk_size: int = 8192, overwrite: bool = False) -> Path:
    """Download a file from `url` to `dest` using streaming (stdlib only)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not overwrite:
        log.info("File already exists at %s, skipping download", dest)
        return dest

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        log.info("Starting download: %s -> %s", url, dest)
        with urllib.request.urlopen(url) as resp, open(tmp_path, "wb") as out_file:
            downloaded = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
        tmp_path.replace(dest)
        log.info("Downloaded %s -> %s (%d bytes)", url, dest, downloaded)
        return dest
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        log.exception("Failed to download %s", url)
        raise


def ensure_pbf_available(url: str, dest: Path, *, overwrite: bool = False) -> Path:
    """Ensure the PBF file exists at `dest`, downloading it from `url` if needed."""
    dest = Path(dest)
    if dest.exists() and not overwrite:
        log.info("PBF already available at %s", dest)
        return dest
    if not url:
        raise ValueError("No URL provided to download PBF file")
    return download_file(url, dest, overwrite=overwrite)


def ensure_pbf_from_config(cfg: Optional[object] = None, *, overwrite: bool = False) -> Path:
    """Read URL/path from config and ensure PBF is present."""
    if cfg is None:
        from src.configs.config import config as cfg

    url = getattr(cfg, "OSM_PBF_URL", "")
    dest = getattr(cfg, "pbf_path", None)
    if callable(dest):
        dest = dest()
    if dest is None:
        dest = Path(getattr(cfg, "RAW_PATH")) / getattr(cfg, "OSM_PBF_NAME")
    return ensure_pbf_available(url, Path(dest), overwrite=overwrite)


