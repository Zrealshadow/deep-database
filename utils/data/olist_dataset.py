import os
import gdown
import tarfile
import pandas as pd
from typing import Optional
from tqdm import tqdm

from relbench.base import Database, Dataset, Table


class OlistDataset(Dataset):
    val_timestamp = pd.Timestamp("2018-03-01")
    test_timestamp = pd.Timestamp("2018-06-01")

    md5hash = "dac3a2b761ceb4c8c9e1d0632b3f21eb"
    download_url = "https://drive.google.com/file/d/13GbyG9XNfhDmmvcLKdx6KjUF_bZ6X-V8/view?usp=sharing"
    refer_url = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data"
    name = "olist"

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir=cache_dir)

    def make_db(self) -> Database:
        r"""Processing the raw files into a database."""
        gdown_cache_root = os.path.join("~", ".cache", "gdown")
        gdown_cache_root = os.path.expanduser(gdown_cache_root)

        output = "olist.tar.xz"
        path = gdown.cached_download(
            OlistDataset.download_url,
            path=os.path.join(gdown_cache_root, output),
            quiet=False,
            fuzzy=True,
            md5=OlistDataset.md5hash
        )

        with tarfile.open(path, 'r:xz') as tar:
            tar.extractall(path=os.path.join(gdown_cache_root))
        print("Extracted files to:", os.path.join(gdown_cache_root))

        data_dir = os.path.join(gdown_cache_root, "olist")

        tables = {}
        table_configs = {
            "customer": {
                "pkey_col": "customer_id",
                "fkey_col_to_pkey_table": {'zip_code_prefix': 'geolocation'},
            },

            "geolocation": {
                "fkey_col_to_pkey_table": {},
                "pkey_col": 'zip_code_prefix',
            },

            "seller": {
                "pkey_col": "seller_id",
                "fkey_col_to_pkey_table": {
                    "zip_code_prefix": "geolocation"
                }
            },

            "payment": {
                "pkey_col": "payment_id",
                "fkey_col_to_pkey_table": {
                    "order_id": "order",
                },
            },

            "order": {
                "pkey_col": "order_id",
                "fkey_col_to_pkey_table": {
                    "customer_id": "customer",
                },
                "time_col": "purchase_time"
            },

            "product": {
                "pkey_col": "product_id",
                "fkey_col_to_pkey_table": {}
            },

            "order_item": {
                "pkey_col": "order_item_id",
                "fkey_col_to_pkey_table": {
                    "order_id": "order",
                    "product_id": "product",
                    "seller_id": "seller"
                }
            },

            "review": {
                "pkey_col": "review_id",
                "fkey_col_to_pkey_table": {
                    "order_id": "order"
                },
                "time_col": "creation_time"
            }
        }

        for table_name, config in tqdm(
            table_configs.items(), desc="Processing tables"
        ):
            df = pd.read_csv(os.path.join(data_dir, f"{table_name}.csv"))

            for col in [
                "purchase_time",
                "approval_time",
                "delivered_carrier_time",
                "delivered_customer_time",
                "estimated_delivery_time",
                "creation_time",
                "answer_time",
                "shipping_limit_date",
            ]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            tables[table_name] = Table(
                df=df,
                **config
            )

        return Database(table_dict=tables)


def _register_olist():
    """Register the Olist dataset."""
    from .database_factory import DatabaseFactory
    from ..task.olist_tasks import OlistOrderDelayTask
    
    def _load_olist_dataset(cache_dir: Optional[str] = None) -> OlistDataset:
        cache_root_dir = os.path.join("~", ".cache", "relbench")
        cache_root_dir = os.path.expanduser(cache_root_dir)
        cache_dir = cache_dir if cache_dir else os.path.join(
            cache_root_dir, "olist")
        return OlistDataset(cache_dir=cache_dir)

    DatabaseFactory.register_dataset("olist", _load_olist_dataset)
    
    # register tasks
    DatabaseFactory.register_task("olist", "order-delay", OlistOrderDelayTask)

_register_olist()
