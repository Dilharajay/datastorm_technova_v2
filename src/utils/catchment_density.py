# Competitive Catchment Density — count competing outlets within a fixed radius

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd

log = logging.getLogger("pipeline.catchment")

DEFAULT_RADIUS_M = 5000
METRIC_CRS = "EPSG:5235"


def compute_competition_density(
    df: pd.DataFrame,
    *,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    id_col: str = "Outlet_ID",
    radius_m: float = DEFAULT_RADIUS_M,
) -> pd.DataFrame:
    """
    For each outlet, count the number of other outlets within `radius_m` metres.

    Uses a buffer-based spatial join: each outlet is buffered by `radius_m`
    and all other outlets whose point falls inside the buffer are counted.

    Returns a DataFrame with ``Outlet_ID`` and ``competition_density`` (integer).
    """
    if df.empty:
        return pd.DataFrame(columns=[id_col, "competition_density"])

    gdf = gpd.GeoDataFrame(
        df[[id_col, lat_col, lon_col]].copy(),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )

    gdf_proj = gdf.to_crs(METRIC_CRS)

    buffers = gdf_proj.copy()
    buffers["geometry"] = buffers.buffer(radius_m)

    joined = gpd.sjoin(
        gdf_proj[["geometry"]],
        buffers[[id_col, "geometry"]],
        how="left",
        predicate="intersects",
    )

    self_match = joined.index == joined["index_right"]
    density = (
        joined[~self_match]
        .groupby(level=0)
        .size()
        .reindex(gdf.index, fill_value=0)
    )

    result = pd.DataFrame({id_col: df[id_col].values, "competition_density": density.values})
    log.info(
        "[Catchment] Competition density computed (radius=%dm) — "
        "mean=%.1f, min=%d, max=%d",
        radius_m,
        density.mean(),
        density.min(),
        density.max(),
    )
    return result
