from typing import Optional
import pandas as pd
from relbench.base import Database, Table, Dataset
from relbench.datasets.avito import AvitoDataset as RDBenchAvitoDataset
import os

from utils.data.event_dataset import preprocess_event_database


def preprocess_avito_database(db: Database):
    # preprocess Avito dataset
    # mainly to solve the Nan values in the foreign key

    # remove VisitStream Nan foreign Keys 'AdID' and 'UserID'
    visit_stream_df = db.table_dict['VisitStream'].df.dropna(
        subset=['AdID', 'UserID'])
    visit_stream_df.reset_index(drop=True, inplace=True)
    db.table_dict['VisitStream'].df = visit_stream_df

    # remove Nan in PhoneRequestsStream
    phone_requests_stream_df = db.table_dict['PhoneRequestsStream'].df.dropna(subset=[
                                                                              'AdID', 'UserID'])
    phone_requests_stream_df.reset_index(drop=True, inplace=True)
    db.table_dict['PhoneRequestsStream'].df = phone_requests_stream_df

    # remove Nan in SearchStream
    search_stream_df = db.table_dict['SearchStream'].df.dropna(
        subset=['AdID', 'SearchID'])
    search_stream_df.reset_index(drop=True, inplace=True)
    db.table_dict['SearchStream'].df = search_stream_df

    # remove Nan in SearchInfo
    search_info_df = db.table_dict['SearchInfo'].df.dropna(
        subset=['SearchID', 'UserID'])
    search_info_df.reset_index(drop=True, inplace=True)
    db.table_dict['SearchInfo'].df = search_info_df

    # Nan value in CategoryID and LocationID in AdsInfo and SearchInfo
    # NOTE: replace those Nan with most frequent value
    most_frequent_location = db.table_dict['SearchInfo'].df['LocationID'].mode()[0]
    most_frequent_category = db.table_dict['SearchInfo'].df['CategoryID'].mode()[0]

    db.table_dict['SearchInfo'].df['LocationID'].fillna(
        most_frequent_location, inplace=True)
    db.table_dict['SearchInfo'].df['CategoryID'].fillna(
        most_frequent_category, inplace=True)

    most_frequent_location = db.table_dict['AdsInfo'].df['LocationID'].mode()[0]
    most_frequent_category = db.table_dict['AdsInfo'].df['CategoryID'].mode()[0]

    db.table_dict['AdsInfo'].df['LocationID'].fillna(
        most_frequent_location, inplace=True)
    db.table_dict['AdsInfo'].df['CategoryID'].fillna(
        most_frequent_category, inplace=True)


class AvitoDataset(RDBenchAvitoDataset):
    """Avito dataset with preprocessing."""

    def make_db(self) -> Database:
        """Create the Database object"""

        # TODO add logic to fetch the raw data
        #
        # ===================================

        db = super().make_db()
        return db

    def get_db(self, upto_test_timestamp=True) -> Database:
        """Reimplement get_db from basic Dataset"""
        """TODO: it's a special case due to historical correctness issuse
            We should move the preprocess logic to make_db part.
            For correctness, we keep current implementation.
            
            After paper submission, need to refactor this part.
        """
        db = super().get_db()
        preprocess_avito_database(db)
        self.validate_and_correct_db(db)
        return db


def _register_avito_dataset():
    from .database_factory import DatabaseFactory
    from relbench.tasks import avito

    def _load_avito_dataset(cache_dir: Optional[str] = None) -> Dataset:
        """Load the Avito dataset."""
        cache_root_dir = os.path.join("~", ".cache", "relbench", "rel-avito")
        cache_root_dir = os.path.expanduser(cache_root_dir)
        cache_dir = cache_dir if cache_dir else cache_root_dir
        return AvitoDataset(cache_dir=cache_dir)

    DatabaseFactory.register_dataset("avito", _load_avito_dataset, None)

    # Register tasks
    DatabaseFactory.register_task("avito", "ad-ctr", avito.AdCTRTask)
    DatabaseFactory.register_task("avito", "user-clicks", avito.UserClicksTask)
    DatabaseFactory.register_task("avito", "user-visits", avito.UserVisitsTask)


# NOTE: temporarily disable Avito dataset registration
# Because it will delete some tuples, which lead to inconsistency with index and pkey
# _register_avito_dataset()
