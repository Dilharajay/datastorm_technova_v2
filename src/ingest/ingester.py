# main ingester script to read raw data and write to bronze layer
# ZIP -> CSV -> extract -> write to bronze in parquet format

import logging
import shutil
import pandas as pd
import zipfile
from pathlib import Path

from src.configs.config import config
from src.utils.io import add_audit_columns, write_parquet

log = logging.getLogger("pipeline.ingester")


class BronzeIngester:
    LAYER = "bronze"

    def run(self) -> dict[str, int]:
        log.info("[%s] Starting ingestion", self.LAYER.upper())
        results: dict[str, int] = {}

        for zip_path in self._discover_zips():
            extract_root = self._extract_zip(zip_path)
            try:
                csv_files = sorted(extract_root.rglob("*.csv"))
                if not csv_files:
                    log.warning(
                        "[%s] No CSV files found in '%s' — skipping",
                        self.LAYER.upper(),
                        zip_path.name,
                    )
                    continue

                log.info(
                    "[%s] ZIP '%s' contains %d CSV file(s): %s",
                    self.LAYER.upper(),
                    zip_path.name,
                    len(csv_files),
                    [f.name for f in csv_files],
                )

                for csv_path in csv_files:
                    table_name = csv_path.stem
                    row_count = self._ingest_csv(csv_path, table_name)
                    results[table_name] = row_count
            finally:
                self._cleanup(extract_root)

        log.info(
            "[%s] Ingestion completed: %d table(s) written",
            self.LAYER.upper(),
            len(results),
        )
        return results

    # Private helpers
    def _discover_zips(self) -> list[Path]:
        files = sorted(config.RAW_PATH.glob(config.raw_file_glob))
        if not files:
            raise FileNotFoundError(
                f"No ZIP files found in '{config.RAW_PATH}'. "
                "Place your source ZIP there and re-run."
            )
        log.info(
            "[%s] Discovered %d ZIP file(s): %s",
            self.LAYER.upper(),
            len(files),
            [f.name for f in files],
        )
        return files

    def _extract_zip(self, zip_path: Path) -> Path:
        extract_root = config.EXTRACT_PATH / zip_path.stem
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True, exist_ok=True)

        log.info(
            "[%s] Extracting '%s' to '%s'",
            self.LAYER.upper(),
            zip_path.name,
            extract_root,
        )
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                dest = (extract_root / member).resolve()
                if not str(dest).startswith(str(extract_root.resolve())):
                    raise ValueError(f"Unsafe path in ZIP: '{member}'")
            zf.extractall(extract_root)

        extracted_items = len(list(extract_root.rglob("*")))
        log.info(
            "[%s] Extracted %d item(s) from '%s'",
            self.LAYER.upper(),
            extracted_items,
            zip_path.name,
        )
        return extract_root

    def _ingest_csv(self, csv_path: Path, table_name: str) -> int:
        df = pd.read_csv(csv_path)
        df = add_audit_columns(df, self.LAYER)
        return write_parquet(df, config.BRONZE_PATH, table_name, self.LAYER)

    def _cleanup(self, extract_root: Path) -> None:
        if extract_root.exists():
            shutil.rmtree(extract_root)
            log.info(
                "[%s] Removed temporary extraction folder: %s",
                self.LAYER.upper(),
                extract_root,
            )
