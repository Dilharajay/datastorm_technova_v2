# main ingester script to read raw data and write to bronze layer
# ZIP -> CSV -> extract -> write to bronze in parquet fprmat

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
        log.info("Starting Bronze Ingester")
        results: dict[str, int] = {}

        for zip_path in self._discover_zips():
            extract_root = self._extract_zip(zip_path)
            try:
                csv_files = sorted(extract_root.rglob("*.csv"))
                if not csv_files:
                    log.warning("No CSVs found inside... %s skipping...", zip_path.name)
                    continue
 
                log.info("  %s  →  %d CSV(s): %s",
                         zip_path.name, len(csv_files), [f.name for f in csv_files])
 
                for csv_path in csv_files:
                    table_name = csv_path.stem
                    row_count  = self._ingest_csv(csv_path, table_name)
                    results[table_name] = row_count
            finally:
                self._cleanup(extract_root)
 
        log.info("Bronze completed!  (%d tables ingested)", len(results))
        return results
 
    # Private helpers 
    def _discover_zips(self) -> list[Path]:
        files = sorted(config.RAW_PATH.glob(config.raw_file_glob))
        if not files:
            raise FileNotFoundError(
                f"No ZIP files found in '{config.RAW_PATH}'. "
                "Place your source ZIP there and re-run."
            )
        log.info("Found %d ZIP file(s): %s", len(files), [f.name for f in files])
        return files
 
    def _extract_zip(self, zip_path: Path) -> Path:
        extract_root = config.EXTRACT_PATH / zip_path.stem
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True, exist_ok=True)
 
        log.info("Extracting  %s  →  %s", zip_path.name, extract_root)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                dest = (extract_root / member).resolve()
                if not str(dest).startswith(str(extract_root.resolve())):
                    raise ValueError(f"Unsafe path in ZIP: '{member}' — aborting.")
            zf.extractall(extract_root)
 
        log.info("  Extracted %d item(s)", len(list(extract_root.rglob("*"))))
        return extract_root
 
    def _ingest_csv(self, csv_path: Path, table_name: str) -> int:
        df = pd.read_csv(csv_path)
        df = add_audit_columns(df, self.LAYER)
        return write_parquet(df, config.BRONZE_PATH, table_name, self.LAYER)
 
    def _cleanup(self, extract_root: Path) -> None:
        if extract_root.exists():
            shutil.rmtree(extract_root)
            log.info("Cleaned up temp folder: %s", extract_root)