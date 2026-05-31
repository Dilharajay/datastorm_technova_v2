# Main cleaning script for silver layer

import logging
from typing import Mapping, Optional

import pandas as pd

from src.configs.config import config
from src.utils.cleaning_utils import (
    add_year_month_from_date,
    filter_numeric_upper_bound,
    flag_flat_volume_outlets,
    flag_zero_volume_rows,
    make_abs_columns,
    normalize_text_column,
    parse_date_column,
)
from src.utils.io import read_parquet, write_parquet

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

        tx = read_parquet(config.BRONZE_PATH, "transactions_history_final")
        log.info("[%s] Loaded '%s' with %d row(s)", self.LAYER.upper(), "transactions_history_final", len(tx))

        out = read_parquet(config.BRONZE_PATH, "outlet_master")
        log.info("[%s] Loaded '%s' with %d row(s)", self.LAYER.upper(), "outlet_master", len(out))

        dist = read_parquet(config.BRONZE_PATH, "distributor_seasonality_details")
        log.info(
            "[%s] Loaded '%s' with %d row(s)",
            self.LAYER.upper(),
            "distributor_seasonality_details",
            len(dist),
        )

        hol = read_parquet(config.BRONZE_PATH, "holiday_list")
        log.info("[%s] Loaded '%s' with %d row(s)", self.LAYER.upper(), "holiday_list", len(hol))

        tx_cleaned = self.clean_transactions(tx)
        log.info("[%s] Cleaned '%s' -> %d row(s)", self.LAYER.upper(), "transactions_history_final", len(tx_cleaned))
        results["transactions_history_final"] = write_parquet(
            tx_cleaned,
            config.SILVER_PATH,
            "transactions_history_final",
            self.LAYER,
        )

        out_cleaned = self.clean_outlet_master(out)
        log.info("[%s] Cleaned '%s' -> %d row(s)", self.LAYER.upper(), "outlet_master", len(out_cleaned))
        results["outlet_master"] = write_parquet(out_cleaned, config.SILVER_PATH, "outlet_master", self.LAYER)

        dist_cleaned = self.clean_distributor_seasonality(dist)
        log.info(
            "[%s] Cleaned '%s' -> %d row(s)",
            self.LAYER.upper(),
            "distributor_seasonality_details",
            len(dist_cleaned),
        )
        results["distributor_seasonality_details"] = write_parquet(
            dist_cleaned,
            config.SILVER_PATH,
            "distributor_seasonality_details",
            self.LAYER,
        )

        hol_cleaned = self.clean_holiday_list(hol)
        log.info("[%s] Cleaned '%s' -> %d row(s)", self.LAYER.upper(), "holiday_list", len(hol_cleaned))
        results["holiday_list"] = write_parquet(hol_cleaned, config.SILVER_PATH, "holiday_list", self.LAYER)

        log.info("[%s] Cleaning completed: %d table(s) written", self.LAYER.upper(), len(results))
        return results

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
    def clean_seasonality_index(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """Normalize seasonality index."""
        clean = df.copy()
        clean[column_name] = clean[column_name].str.lower().astype("category")
        clean[column_name].unique()
        return clean