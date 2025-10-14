import os
import gdown
import pandas as pd
import tarfile
from typing import Optional
from tqdm import tqdm

from relbench.base import Database, Dataset, Table


class RateBeerDataset(Dataset):
    val_timestamp = pd.Timestamp("2023-06-01")
    test_timestamp = pd.Timestamp("2024-06-01")

    DOWNLOAD_LINK = "https://drive.google.com/file/d/1r3e7T6kA-ImCCkIc029FHiMBj6QT8iF2/view?usp=sharing"
    # already cleaned

    DB_URL = "https://www.dropbox.com/scl/fi/exwygxep7vdvq55uiq28r/db.zip?rlkey=o7q0r8nw758p4wxx1wka9ubuj&st=rg3gvkxg&dl=1"
    # raw

    md5hash = "4cd20216af99caa18535a10d4731f450"

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir=cache_dir)

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""
        gdown_cache_root = os.path.join("~", ".cache", "gdown")
        gdown_cache_root = os.path.expanduser(gdown_cache_root)

        output = "ratebeer.tar.xz"
        path = gdown.cached_download(
            RateBeerDataset.DOWNLOAD_LINK,
            path=os.path.join(gdown_cache_root, output),
            quiet=False,
            fuzzy=True,
            md5=RateBeerDataset.md5hash
        )
        with tarfile.open(path, 'r:xz') as tar:
            tar.extractall(path=os.path.join(gdown_cache_root))
        print("Extracted files to:", os.path.join(gdown_cache_root))
        data_dir = os.path.join(gdown_cache_root, "ratebeer")

        print("Reading from processed database...")
        tables = {}

        # Define table configurations
        table_configs = {
            # Reference tables
            "countries": {"pkey": "country_id", "fkeys": {}, "time_col": None},

            # Core tables
            "places": {
                "pkey": "place_id",
                "fkeys": {
                    "country_id": "countries",
                },
                "time_col": "created_at"
            },
            "beers": {
                "pkey": "beer_id",
                "fkeys": {
                    "brewer_id": "brewers",
                },
                "time_col": "created_at"
            },
            "beer_ratings": {
                "pkey": "rating_id",
                "fkeys": {
                    "beer_id": "beers",
                    "user_id": "users"
                },
                "time_col": "created_at"
            },
            "availability": {
                "pkey": "avail_id",
                "fkeys": {
                    "beer_id": "beers",
                    "country_id": "countries",
                    "place_id": "places"
                },
                "time_col": "created_at"
            },
            "favorites": {
                "pkey": "favorite_id",
                "fkeys": {
                    "beer_id": "beers",
                    "user_id": "users"
                },
                "time_col": "created_at"
            },
            "place_ratings": {
                "pkey": "rating_id",
                "fkeys": {
                    "place_id": "places",
                    "user_id": "users"
                },
                "time_col": "created_at"
            },
            "brewers": {
                "pkey": "brewer_id",
                "fkeys": {"country_id": "countries"},
                "time_col": "created_at"
            },
            # Users table combines information about users who rate beers and places
            "users": {
                "pkey": "user_id",
                "fkeys": {},
                "time_col": "created_at"
            }
        }

        # Read tables from extracted directory
        for table_name, config in tqdm(table_configs.items(), desc="Loading tables", total=len(table_configs), leave=False):
            csv_path = os.path.join(data_dir, f"{table_name}.csv")
            df = pd.read_csv(csv_path, low_memory=False)

            # === Start Modification: Handle potential duplicates in primary keys ===
            pkey_col = config.get("pkey")
            # Check only for single-column primary keys (strings)
            if isinstance(pkey_col, str):
                initial_rows = len(df)
                if df.duplicated(subset=pkey_col).any():
                    num_duplicates = df.duplicated(subset=pkey_col).sum()
                    print(
                        f"\nWarning: Found {num_duplicates} duplicate value(s) for primary key '{pkey_col}' in {table_name}.csv. Removing duplicates, keeping first occurrence.")
                    df = df.drop_duplicates(subset=pkey_col, keep="first")
                    print(
                        f"Removed {initial_rows - len(df)} rows from {table_name}. New shape: {df.shape}")
            # Note: Composite keys (lists) and pkey=None are not checked here.
            # The base library issue with composite key re-indexing still exists,
            # so beer_aliases pkey remains set to None in table_configs as a workaround.
            # === End Modification ===

            # Convert timestamp columns if present
            if config["time_col"] is not None:
                # Store the time column name
                time_col_name = config["time_col"]
                for col in ["created_at", "updated_at", "last_edited_at", "opened_at"]:
                    if col in df.columns:
                        # Use errors='coerce' to turn unparseable dates into NaT
                        df[col] = pd.to_datetime(
                            df[col], format='mixed', errors='coerce')

                # === Start Modification: Remove rows with NaT in the designated time_col ===
                if time_col_name in df.columns and df[time_col_name].isna().any():
                    initial_rows = len(df)
                    nat_count = df[time_col_name].isna().sum()
                    print(
                        f"\nWarning: Found {nat_count} NaT value(s) in time column '{time_col_name}' for table '{table_name}'. Removing these rows.")
                    df = df.dropna(subset=[time_col_name])
                    print(
                        f"Removed {initial_rows - len(df)} rows from {table_name}. New shape: {df.shape}")
                # === End Modification ===

            tables[table_name] = Table(
                df=df,
                fkey_col_to_pkey_table=config["fkeys"],
                pkey_col=config["pkey"],
                time_col=config["time_col"]
            )
            tqdm.write(f"Loaded {table_name}: {len(df):,} rows")

        # rm temporary attributes, which related to a duration of time
        # tables["beers"].df.drop(
        #     columns=[
        #         'year4_avg',
        #         'year4_overall',
        #         'year4_count',
        #         'last_9m_avg',
        #         'last_9m_count',
        #     ],
        #     inplace=True,
        # )

        print("\nAll tables loaded successfully!")
        return Database(tables)


# ============================================================================
# Self-registration with DatabaseFactory
# ============================================================================

def _register_ratebeer():
    """Register RateBeer dataset and tasks with DatabaseFactory."""
    from .database_factory import DatabaseFactory
    from utils.task import ActiveUserPredictionTask, BeerPositiveRatePredictionTask, PlacePositivePredictionTask

    def _load_ratebeer_dataset(cache_dir: Optional[str] = None) -> Dataset:
        """Load the RateBeer dataset."""
        cache_root_dir = os.path.join("~", ".cache", "relbench")
        cache_root_dir = os.path.expanduser(cache_root_dir)
        cache_dir = cache_dir if cache_dir else os.path.join(
            cache_root_dir, "ratebeer")
        return RateBeerDataset(cache_dir=cache_dir)

    # Register dataset
    DatabaseFactory.register_dataset("ratebeer", _load_ratebeer_dataset)

    # Register tasks
    DatabaseFactory.register_task(
        "ratebeer", "user-active", ActiveUserPredictionTask)
    DatabaseFactory.register_task(
        "ratebeer", "beer-positive", BeerPositiveRatePredictionTask)
    DatabaseFactory.register_task(
        "ratebeer", "place-positive", PlacePositivePredictionTask)


# Auto-register when this module is imported
_register_ratebeer()
