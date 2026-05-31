# main configuration file for the project

from dataclasses import dataclass, field
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent.parent

@dataclass
class Config:

    # data layer paths
    RAW_PATH: Path = BASE_PATH / "data" / "raw" 
    BRONZE_PATH: Path = BASE_PATH / "data" / "bronze"
    SILVER_PATH: Path = BASE_PATH / "data" / "silver"
    GOLD_PATH: Path = BASE_PATH / "data" / "gold"
    EXTRACT_PATH: Path = BASE_PATH / "data" / "extracted" # temporary path for extracted files, auto cleaned

    # write options
    parquet_compression: str = "snappy"

    # source file dicovery
    raw_file_glob: str = "*.zip" # zip containing csv files

    # audit column names
    col_layer: str = "_layer"
    col_loaded_at: str = "_loaded_at"
    col_pipeline_run: str = "_pipeline_run"

    def managed_dirs(self) -> tuple:
        # all directories that should be exist before running the pipeline
        return (self.RAW_PATH, self.BRONZE_PATH, self.SILVER_PATH, self.GOLD_PATH)

# create a global config instance
config = Config()
