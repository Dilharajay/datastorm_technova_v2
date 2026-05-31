# main configuration file for the project

from dataclasses import dataclass
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent.parent

@dataclass
class Config:

    # data layer paths
    RAW_PATH: Path = BASE_PATH / "data" / "raw" 
    BRONZE_PATH: Path = BASE_PATH / "data" / "bronze"
    SILVER_PATH: Path = BASE_PATH / "data" / "silver"
    GOLD_PATH: Path = BASE_PATH / "data" / "gold"
    REJECTS_PATH: Path = BASE_PATH / "data" / "rejects"
    EXTRACT_PATH: Path = BASE_PATH / "data" / "extracted" # temporary path for extracted files, auto cleaned
    FIGURES_DIR: Path = BASE_PATH / "figures"
    REPORTS_DIR: Path = BASE_PATH / "reports"
    # optional OSM PBF download configuration
    OSM_PBF_URL: str = "https://download.geofabrik.de/asia/sri-lanka-latest.osm.pbf"
    # filename to store the downloaded pbf under RAW_PATH
    OSM_PBF_NAME: str = "sri-lanka-latest.osm.pbf"
    """Alt name sri-lanka-latest.osm.pbf"""

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
        return (self.RAW_PATH, self.BRONZE_PATH, self.SILVER_PATH, self.GOLD_PATH, self.REJECTS_PATH)

    def pbf_path(self) -> Path:
        """Return the expected path for the configured OSM PBF file."""
        return self.RAW_PATH / self.OSM_PBF_NAME

# create a global config instance
config = Config()
