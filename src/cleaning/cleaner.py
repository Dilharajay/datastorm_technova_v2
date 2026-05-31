# Main cleaning script for silver layer

from pathlib import Path
import logging
from typing import Mapping, Optional

import geopandas as gpd
import pandas as pd

from src.configs.config import config
from src.utils.cleaning_utils import (
    add_year_month_from_date,
    clean_and_verify_outlet_coordinates,
    clean_outlet_coordinates_basic,
    ensure_pbf_from_config,
    filter_numeric_upper_bound,
    flag_flat_volume_outlets,
    flag_zero_volume_rows,
    make_abs_columns,
    normalize_text_column,
    parse_date_column,
)
from src.utils.io import drop_audit_columns, read_parquet, write_parquet, write_rejects
from src.utils.poi_utils import POI_CATEGORIES, compute_outlet_poi_scores, extract_all_poi_categories

log = logging.getLogger("pipeline.cleaner")

DEFAULT_OUTLET_TYPE_MAPPING = {
    "grocry": "grocery",
    "grocery": "grocery",
    "bakry": "bakery",
    "bakery": "bakery",
    "hotel": "hotel",
    "pharmacy": "pharmacy",
    "kiosk": "kiosk",
    "eatery": "eatery",
    "smmt": "smmt",
}


class SilverCleaner:
    LAYER = "silver"

    def run(self) -> dict[str, int]:
        log.info("[%s] Starting cleaning", self.LAYER.upper())
        results: dict[str, int] = {}
        pbf_path = None

        # ensure optional OSM PBF is present (download if configured)
        try:
            pbf_path = ensure_pbf_from_config()
            log.info("OSM PBF ensured at %s", pbf_path)
        except ValueError:
            log.debug("No OSM PBF URL configured; skipping download")
        except Exception as exc:
            log.warning("Failed to ensure OSM PBF file: %s", exc)

        # ── Transactions ──────────────────────────────────────────────
        tx = read_parquet(config.BRONZE_PATH, "transactions_history_final")
        log.info("[%s] Loaded '%s' with %d row(s)", self.LAYER.upper(), "transactions_history_final", len(tx))

        tx_valid_raw, tx_reject_df = self._reject_transactions(tx)

        tx_cleaned = self.clean_transactions(tx_valid_raw)
        tx_cleaned = tx_cleaned.drop(columns=["flag_zero_volume", "flag_flat_outlet"], errors="ignore")

        log.info("[%s] Cleaned '%s' -> %d valid, %d rejected",
                 self.LAYER.upper(), "transactions_history_final", len(tx_cleaned), len(tx_reject_df))
        results["transactions_history_final"] = write_parquet(
            tx_cleaned, config.SILVER_PATH, "transactions_history_final", self.LAYER,
        )
        if not tx_reject_df.empty:
            write_rejects(tx_reject_df, config.REJECTS_PATH, "transactions_history_final", self.LAYER)

        # ── Outlet Master ─────────────────────────────────────────────
        out = read_parquet(config.BRONZE_PATH, "outlet_master")
        log.info("[%s] Loaded '%s' with %d row(s)", self.LAYER.upper(), "outlet_master", len(out))

        out_cleaned = self.clean_outlet_master(out)
        log.info("[%s] Cleaned '%s' -> %d row(s)", self.LAYER.upper(), "outlet_master", len(out_cleaned))
        results["outlet_master"] = write_parquet(out_cleaned, config.SILVER_PATH, "outlet_master", self.LAYER)

        # ── Distributor Seasonality ───────────────────────────────────
        dist = read_parquet(config.BRONZE_PATH, "distributor_seasonality_details")
        log.info("[%s] Loaded '%s' with %d row(s)", self.LAYER.upper(), "distributor_seasonality_details", len(dist))

        dist_cleaned = self.clean_distributor_seasonality(dist)
        log.info("[%s] Cleaned '%s' -> %d row(s)", self.LAYER.upper(), "distributor_seasonality_details", len(dist_cleaned))
        results["distributor_seasonality_details"] = write_parquet(
            dist_cleaned, config.SILVER_PATH, "distributor_seasonality_details", self.LAYER,
        )

        # ── Holiday List ──────────────────────────────────────────────
        hol = read_parquet(config.BRONZE_PATH, "holiday_list")
        log.info("[%s] Loaded '%s' with %d row(s)", self.LAYER.upper(), "holiday_list", len(hol))

        hol_cleaned = self.clean_holiday_list(hol)

        # Reject rows with unparseable dates
        nat_mask = hol_cleaned["Date"].isna()
        hol_reject = hol_cleaned.loc[nat_mask].copy() if nat_mask.any() else pd.DataFrame()

        # Deduplicate remaining
        hol_valid = hol_cleaned.loc[~nat_mask].drop_duplicates(
            subset=["Date", "Holiday_Name", "Holiday_Type"]
        ).copy()

        log.info("[%s] Cleaned '%s' -> %d valid, %d rejected (bad dates + dupes)",
                 self.LAYER.upper(), "holiday_list", len(hol_valid),
                 len(hol_reject) + len(hol_cleaned) - len(hol_valid) - len(hol_reject))
        results["holiday_list"] = write_parquet(hol_valid, config.SILVER_PATH, "holiday_list", self.LAYER)
        if not hol_reject.empty:
            write_rejects(hol_reject, config.REJECTS_PATH, "holiday_list", self.LAYER, reject_reason="unparseable_date")

        # ── Outlet Coordinates ────────────────────────────────────────
        geo = read_parquet(config.BRONZE_PATH, "outlet_coordinates")
        log.info("[%s] Loaded '%s' with %d row(s)", self.LAYER.upper(), "outlet_coordinates", len(geo))

        geo_valid, geo_reject = self.clean_outlet_coordinates(geo, pbf_path=pbf_path)
        log.info("[%s] Cleaned '%s' -> %d valid, %d rejected",
                 self.LAYER.upper(), "outlet_coordinates", len(geo_valid), len(geo_reject))
        results["outlet_coordinates"] = write_parquet(geo_valid, config.SILVER_PATH, "outlet_coordinates", self.LAYER)
        if not geo_reject.empty:
            write_rejects(geo_reject, config.REJECTS_PATH, "outlet_coordinates", self.LAYER)

        # ── POI Scores (uses deduplicated geo_valid) ──────────────────
        poi_results = self.create_outlet_poi_scores(geo_valid, pbf_path=pbf_path)
        results.update(poi_results)

        log.info("[%s] Cleaning completed: %d table(s) written, %d rejected",
                 self.LAYER.upper(), len(results),
                 len(tx_reject_df) + (0 if hol_reject.empty else len(hol_reject)) + len(geo_reject))
        return results

    @staticmethod
    def _reject_transactions(
        df: pd.DataFrame,
        max_volume: float = 8000,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identify rows in the raw bronze transactions that fail validation
        *before* cleaning (negatives, zero volume, over-max volume).
        Returns ``(valid_df, reject_df)`` where ``reject_df`` has a
        ``_reject_reason`` column.
        """
        neg_vol = df["Volume_Liters"] < 0
        neg_bill = df["Total_Bill_Value"] < 0
        zero_vol = df["Volume_Liters"] == 0
        over_max = df["Volume_Liters"] >= max_volume
        reject_mask = neg_vol | neg_bill | zero_vol | over_max

        def _reason(row: pd.Series) -> str:
            parts = []
            if row["Volume_Liters"] < 0:
                parts.append("negative_volume")
            if row["Total_Bill_Value"] < 0:
                parts.append("negative_bill")
            if row["Volume_Liters"] == 0:
                parts.append("zero_volume")
            if row["Volume_Liters"] >= max_volume:
                parts.append("volume_exceeds_max")
            return ";".join(parts)

        reject = df.loc[reject_mask].copy()
        reject["_reject_reason"] = reject.apply(_reason, axis=1)

        valid = df.loc[~reject_mask].copy()
        return valid, reject

    @staticmethod
    def clean_transactions(
        df: pd.DataFrame,
        *,
        max_volume_liters: float = 8000,
        volume_col: str = "Volume_Liters",
        bill_col: str = "Total_Bill_Value",
        outlet_col: str = "Outlet_ID",
        add_flags: bool = True,
        min_months_for_flat_flag: int = 3,
        flat_tolerance: float = 0.01,
    ) -> pd.DataFrame:
        """
        Apply the transaction cleaning logic used in the notebook:
        - absolute value for numeric amount columns
        - remove extreme Volume_Liters rows
        - add zero-volume and flat-pattern flags
        """
        clean = df.copy()

        clean = make_abs_columns(clean, [volume_col, bill_col], copy=False)
        clean = filter_numeric_upper_bound(clean, volume_col, max_volume_liters, copy=False, inclusive=False)

        if add_flags:
            clean["flag_zero_volume"] = flag_zero_volume_rows(clean, volume_col=volume_col)
            clean["flag_flat_outlet"] = flag_flat_volume_outlets(
                clean,
                outlet_col=outlet_col,
                volume_col=volume_col,
                min_months=min_months_for_flat_flag,
                tolerance=flat_tolerance,
            )

        return clean

    @staticmethod
    def clean_outlet_master(
        df: pd.DataFrame,
        *,
        size_fill_strategy: str = "UNKNOWN",
        size_fill_value: str = "UNKNOWN",
        type_mapping: Optional[Mapping[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Apply outlet_master cleaning logic:
        - normalize Outlet_Size
        - mark imputed rows
        - fill missing Outlet_Size
        - normalize Outlet_Type using a mapping
        """
        clean = df.copy()
        mapping = dict(DEFAULT_OUTLET_TYPE_MAPPING)
        if type_mapping is not None:
            mapping.update(type_mapping)

        clean = normalize_text_column(clean, "Outlet_Size", lower=True)
        if "Outlet_Size" in clean.columns:
            clean["Outlet_Size_Imputed"] = clean["Outlet_Size"].isna()
            if size_fill_strategy.lower() == "mode":
                non_null = clean["Outlet_Size"].dropna()
                fill_value = non_null.mode().iloc[0] if not non_null.empty else size_fill_value.lower()
            else:
                fill_value = size_fill_value.lower()
            clean["Outlet_Size"] = clean["Outlet_Size"].fillna(fill_value)
            clean["Outlet_Size"] = clean["Outlet_Size"].astype("object")

        if "Outlet_Type" in clean.columns:
            normalized = clean["Outlet_Type"].astype("string").str.strip().str.lower()
            clean["Outlet_Type"] = normalized.map(mapping).fillna(normalized).astype("object")

        return clean

    @staticmethod
    def clean_distributor_seasonality(
        df: pd.DataFrame,
        *,
        category_col: str = "Seasonality_Index",
    ) -> pd.DataFrame:
        """Normalize the distributor seasonality table."""
        clean = df.copy()
        if category_col in clean.columns:
            clean[category_col] = (
                clean[category_col]
                .astype("string")
                .str.strip()
                .str.lower()
                .astype("object")
            )
        return clean

    @staticmethod
    def clean_holiday_list(
        df: pd.DataFrame,
        *,
        date_col: str = "Date",
        name_col: str = "Holiday_Name",
        type_col: str = "Holiday_Type",
        year_col: str = "Year",
        month_col: str = "Month",
    ) -> pd.DataFrame:
        """Normalize holiday names/types and derive Year/Month from Date."""
        clean = df.copy()
        clean = normalize_text_column(clean, name_col, lower=True, collapse_whitespace=True)
        clean = normalize_text_column(clean, type_col, lower=True, collapse_whitespace=True)
        clean = parse_date_column(clean, date_col)
        clean = add_year_month_from_date(clean, date_col, year_col, month_col)
        return clean

    @staticmethod
    def clean_outlet_coordinates(
        df: pd.DataFrame,
        *,
        pbf_path: Optional[Path] = None,
        lat_col: str = "Latitude",
        lon_col: str = "Longitude",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply notebook geo cleaning and return ``(valid_df, reject_df)``.

        - Rows with null lat/lon → reject ``"missing_coordinates"``
        - Rows outside Sri Lanka bounds → reject ``"coordinates_out_of_bounds"``
        - Surviving rows are deduplicated by ``Outlet_ID`` and spatial-join
          artifact columns are dropped.
        """
        reject_frames: list[pd.DataFrame] = []

        # Step 1 — separate rows with missing coordinates
        has_coords = df[lat_col].notna() & df[lon_col].notna()
        missing = df.loc[~has_coords].copy()
        if not missing.empty:
            missing["_reject_reason"] = "missing_coordinates"
            reject_frames.append(missing)

        to_clean = df.loc[has_coords].copy()

        # Step 2 — run existing cleaning pipeline
        if pbf_path:
            try:
                cleaned = clean_and_verify_outlet_coordinates(
                    to_clean, pbf_path, lat_col=lat_col, lon_col=lon_col,
                )
            except FileNotFoundError:
                log.warning("PBF not found; falling back to basic coordinate cleaning")
                cleaned = clean_outlet_coordinates_basic(to_clean, lat_col=lat_col, lon_col=lon_col)
            except Exception as exc:
                log.warning("Outlet boundary verification skipped: %s", exc)
                cleaned = clean_outlet_coordinates_basic(to_clean, lat_col=lat_col, lon_col=lon_col)
        else:
            cleaned = clean_outlet_coordinates_basic(to_clean, lat_col=lat_col, lon_col=lon_col)

        # Step 3 — rows that disappeared are out-of-bounds
        dropped_idx = to_clean.index.difference(cleaned.index)
        if not dropped_idx.empty:
            oob = to_clean.loc[dropped_idx].copy()
            oob["_reject_reason"] = "coordinates_out_of_bounds"
            reject_frames.append(oob)

        # Step 4 — deduplicate by Outlet_ID, drop spatial-join artifacts
        artifact_cols = ["index_right", "name", "admin_level"]
        cleaned = cleaned.drop(columns=[c for c in artifact_cols if c in cleaned.columns], errors="ignore")
        valid = cleaned.drop_duplicates(subset=["Outlet_ID"]).copy()

        reject = pd.concat(reject_frames, ignore_index=True) if reject_frames else pd.DataFrame()
        return valid, reject

    @staticmethod
    def create_outlet_poi_scores(
        geo_cleaned: pd.DataFrame,
        *,
        pbf_path: Optional[Path] = None,
        lam: float = 150,
    ) -> dict[str, int]:
        """
        Compute distance-decayed POI accessibility scores for every outlet
        and write the result as a new silver table ``outlet_poi_scores``.

        Returns a dict ``{"outlet_poi_scores": row_count}`` (or empty if
        the PBF is unavailable).
        """
        if pbf_path is None or not pbf_path.exists():
            log.info("[POI] No PBF available; skipping POI score computation")
            return {}

        log.info("[POI] Computing outlet POI scores from %s", pbf_path)

        uniq = geo_cleaned.drop_duplicates(subset=["Outlet_ID"]).copy()
        outlet_gdf = gpd.GeoDataFrame(
            uniq,
            geometry=gpd.points_from_xy(uniq["Longitude"], uniq["Latitude"]),
            crs="EPSG:4326",
        )

        log.info("[POI] Extracting all POI categories in a single PBF pass ...")
        poi_dict = extract_all_poi_categories(pbf_path)

        scores_per_cat: dict[str, pd.Series] = {}
        for cat in POI_CATEGORIES:
            pois = poi_dict[cat]
            cat_scores = compute_outlet_poi_scores(outlet_gdf, pois, lam=lam)
            scores_per_cat[f"{cat}_score"] = cat_scores
            log.info(
                "[POI] %s score — min=%.4f, mean=%.4f, max=%.4f",
                cat,
                cat_scores.min(),
                cat_scores.mean(),
                cat_scores.max(),
            )

        scores_df = pd.DataFrame(scores_per_cat)
        scores_df["Outlet_ID"] = uniq["Outlet_ID"].values
        scores_df = scores_df[["Outlet_ID"] + list(scores_per_cat.keys())]

        row_count = write_parquet(scores_df, config.SILVER_PATH, "outlet_poi_scores", "silver")
        log.info("[POI] Computed scores for %d unique outlets", len(uniq))
        log.info("[POI] Wrote %d outlet POI score row(s)", row_count)
        return {"outlet_poi_scores": row_count}


class GoldCleaner:
    LAYER = "gold"

    def run(self) -> dict[str, int]:
        log.info("[%s] Starting gold layer build", self.LAYER.upper())

        tx = read_parquet(config.SILVER_PATH, "transactions_history_final")
        log.info("[%s] Loaded '%s' with %d row(s)", self.LAYER.upper(), "transactions_history_final", len(tx))

        out = read_parquet(config.SILVER_PATH, "outlet_master")
        log.info("[%s] Loaded '%s' with %d row(s)", self.LAYER.upper(), "outlet_master", len(out))

        geo = read_parquet(config.SILVER_PATH, "outlet_coordinates")
        log.info("[%s] Loaded '%s' with %d row(s)", self.LAYER.upper(), "outlet_coordinates", len(geo))

        dist = read_parquet(config.SILVER_PATH, "distributor_seasonality_details")
        log.info(
            "[%s] Loaded '%s' with %d row(s)",
            self.LAYER.upper(),
            "distributor_seasonality_details",
            len(dist),
        )

        hol = read_parquet(config.SILVER_PATH, "holiday_list")
        log.info("[%s] Loaded '%s' with %d row(s)", self.LAYER.upper(), "holiday_list", len(hol))

        poi = None
        try:
            poi = read_parquet(config.SILVER_PATH, "outlet_poi_scores")
            log.info("[%s] Loaded '%s' with %d row(s)", self.LAYER.upper(), "outlet_poi_scores", len(poi))
        except Exception:
            log.info("[%s] 'outlet_poi_scores' not available; skipping POI enrichment", self.LAYER.upper())

        gold = self.build_fact_table(tx, out, geo, dist, hol, poi_scores=poi)
        log.info("[%s] Built fact table with %d rows x %d cols", self.LAYER.upper(), *gold.shape)

        results: dict[str, int] = {}
        results["fact_table"] = write_parquet(gold, config.GOLD_PATH, "fact_table", self.LAYER)
        log.info("[%s] Gold layer build completed (1 table written)", self.LAYER.upper())
        return results

    @staticmethod
    def build_fact_table(
        transactions: pd.DataFrame,
        outlet_master: pd.DataFrame,
        outlet_coordinates: pd.DataFrame,
        distributor_seasonality: pd.DataFrame,
        holiday_list: pd.DataFrame,
        poi_scores: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        fact = drop_audit_columns(transactions.copy())

        om = drop_audit_columns(outlet_master.copy())
        fact = fact.merge(om, on="Outlet_ID", how="left")

        geo = drop_audit_columns(outlet_coordinates.copy())
        geo = geo.drop_duplicates(subset=["Outlet_ID"]).copy()
        fact = fact.merge(
            geo[["Outlet_ID", "Latitude", "Longitude", "coord_status"]],
            on="Outlet_ID",
            how="inner",
        )

        ds = drop_audit_columns(distributor_seasonality.copy())
        fact = fact.merge(ds, on=["Distributor_ID", "Year", "Month"], how="left")

        hl = drop_audit_columns(holiday_list.copy())
        hol_agg = (
            hl.groupby(["Year", "Month"], as_index=False)
            .agg(
                holiday_count=("Holiday_Name", "count"),
                holiday_names=("Holiday_Name", lambda x: ", ".join(x.dropna().unique())),
            )
        )
        fact = fact.merge(hol_agg, on=["Year", "Month"], how="left")

        if poi_scores is not None:
            ps = drop_audit_columns(poi_scores.copy())
            fact = fact.merge(ps, on="Outlet_ID", how="left")

        return fact