# POI extraction and distance-decay scoring from OSM PBF data

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Non closed ring detected", category=RuntimeWarning)

log = logging.getLogger("pipeline.poi")

POI_CATEGORIES: dict[str, dict[str, list[str]]] = {
    "school": {
        "amenity": ["school", "university", "college", "kindergarten"],
    },
    "hospital": {
        "amenity": ["hospital", "clinic", "doctors"],
    },
    "bus_stop": {
        "highway": ["bus_stop"],
        "amenity": ["bus_station"],
    },
    "tourist": {
        "tourism": ["attraction", "viewpoint", "museum", "monument"],
    },
}

HARD_CUTOFF_M = 1000
DEFAULT_LAMBDA = 150


def _matches_category(row: pd.Series, tag_filters: dict[str, list[str]]) -> bool:
    for key, values in tag_filters.items():
        val = row.get(key)
        if val is not None and val in values:
            return True
    return False


def extract_all_poi_categories(
    pbf_path: str | Path,
    *,
    layers: tuple[str, ...] = ("points", "multipolygons"),
) -> dict[str, gpd.GeoDataFrame]:
    """
    Read an OSM PBF **once** and extract POIs for all categories defined
    in ``POI_CATEGORIES``.

    Returns a dict ``{category_name: GeoDataFrame}``.
    """
    pbf_path = Path(pbf_path)
    if not pbf_path.exists():
        raise FileNotFoundError(f"PBF file not found: {pbf_path}")

    result: dict[str, list[gpd.GeoDataFrame]] = {cat: [] for cat in POI_CATEGORIES}

    for layer in layers:
        try:
            gdf = gpd.read_file(pbf_path, layer=layer)
        except Exception:
            continue

        is_poly = layer in ("multipolygons", "polygons", "multilinestrings")

        for cat, tag_filters in POI_CATEGORIES.items():
            mask = gdf.apply(_matches_category, axis=1, tag_filters=tag_filters)
            matched = gdf.loc[mask].copy()
            if not matched.empty:
                if is_poly:
                    matched = matched.to_crs("EPSG:5235")
                    matched["geometry"] = matched.centroid
                    matched = matched.to_crs("EPSG:4326")
                result[cat].append(matched)

    output: dict[str, gpd.GeoDataFrame] = {}
    for cat in POI_CATEGORIES:
        if result[cat]:
            combined = pd.concat(result[cat], ignore_index=True)
            if "osm_id" in combined.columns:
                combined = combined.drop_duplicates(subset=["osm_id"]).reset_index(drop=True)
            output[cat] = combined
        else:
            output[cat] = gpd.GeoDataFrame({"osm_id": []}, geometry=[], crs="EPSG:4326")
        log.info("[POI] Extracted %d '%s' POI(s)", len(output[cat]), cat)

    return output


def compute_outlet_poi_scores(
    outlet_gdf: gpd.GeoDataFrame,
    poi_gdf: gpd.GeoDataFrame,
    *,
    lam: float = DEFAULT_LAMBDA,
    hard_cutoff_m: float = HARD_CUTOFF_M,
) -> pd.Series:
    """
    For each outlet, compute the exponential distance-decay score:

        score = sum max( exp(-d_i / lam), 0 )

    where d_i is the distance (metres) to each POI.
    POIs farther than ``hard_cutoff_m`` contribute 0.

    Returns a Series indexed by ``Outlet_ID``.
    """
    if poi_gdf.empty:
        return pd.Series(0.0, index=outlet_gdf["Outlet_ID"], name="score")

    METRIC_CRS = "EPSG:5235"
    o_proj = outlet_gdf.to_crs(METRIC_CRS)[["Outlet_ID", "geometry"]].copy()
    p_proj = poi_gdf.to_crs(METRIC_CRS)[["geometry"]].copy()

    joined = o_proj.sjoin_nearest(
        p_proj,
        how="left",
        max_distance=hard_cutoff_m,
        distance_col="dist_m",
    )

    joined["weight"] = np.exp(-joined["dist_m"] / lam)
    scores = joined.groupby("Outlet_ID")["weight"].sum()

    all_ids = outlet_gdf["Outlet_ID"]
    scores = scores.reindex(all_ids, fill_value=0.0)
    return scores
