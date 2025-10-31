from typing import Optional
import pandas as pd
from relbench.base import Database, Table, Dataset
from relbench.datasets.event import EventDataset as RDBenchEventDataset
import os


def preprocess_event_database(db: Database):
    # ------------ Preprocess some types of data ------------
    # -> <event_attendess>
    # drop nan values in event and user_id
    event_attendees_flattened_df = db.table_dict["event_attendees"].df
    event_attendees_flattened_df = event_attendees_flattened_df.dropna(subset=[
                                                                       'event', 'user_id'])
    # transfer Unname:0 to id and reindex this column
    event_attendees_flattened_df = event_attendees_flattened_df.rename(
        columns={'Unnamed: 0': 'id'}).reset_index(drop=True)
    event_attendees_flattened_df['id'] = event_attendees_flattened_df.index

    # -> <user_friends>
    # drop nan values in user and friend\
    user_friends_flattened_df = db.table_dict["user_friends"].df
    user_friends_flattened_df = user_friends_flattened_df.dropna(subset=[
                                                                 'user', 'friend'])
    # transfer Unname:0 to id and reindex this column
    user_friends_flattened_df = user_friends_flattened_df.rename(
        columns={'Unnamed: 0': 'id'}).reset_index(drop=True)
    user_friends_flattened_df['id'] = user_friends_flattened_df.index

    # -> <event_interest>
    # drop nan values in event and user
    event_interest_df = db.table_dict["event_interest"].df.copy()
    event_interest_df = event_interest_df.dropna(subset=['event', 'user'])
    # add a new id as primaryKey
    event_interest_df.reset_index(drop=True, inplace=True)
    event_interest_df['id'] = event_interest_df.index

    # -> event,

    # collect the event_id which occurs in event_interest and event_attendees.
    event_interest_event_id = set(event_interest_df['event'].unique())
    event_attendees_event_id = set(
        event_attendees_flattened_df['event'].unique())
    involved_event_id = event_interest_event_id | event_attendees_event_id

    event_df = db.table_dict["events"].df
    event_df = event_df[event_df['event_id'].isin(involved_event_id)]

    # reindex the event_id
    event_df.reset_index(drop=True, inplace=True)
    event_id2index = {event_id: index for index,
                      event_id in enumerate(event_df['event_id'])}

    event_df.replace({"event_id": event_id2index}, inplace=True)
    event_interest_df.replace({"event": event_id2index}, inplace=True)
    event_attendees_flattened_df.replace(
        {"event": event_id2index}, inplace=True)

    # event_df["event_id"].replace(event_id2index, inplace=True)
    # # map the event_id in event_interest and event_attendees
    # event_interest_df["event"].replace(event_id2index, inplace=True)
    # event_attendees_flattened_df["event"].replace(event_id2index, inplace=True)

    # reset the table
    db.table_dict["event_attendees"] = Table(
        df=event_attendees_flattened_df,
        fkey_col_to_pkey_table={
            "event": "events",
            "user_id": "users",
        },
        pkey_col="id",
        time_col="start_time",
    )

    db.table_dict["event_interest"] = Table(
        df=event_interest_df,
        fkey_col_to_pkey_table={
            "event": "events",
            "user": "users",
        },
        pkey_col="id",
        time_col="timestamp",
    )

    db.table_dict["user_friends"] = Table(
        df=user_friends_flattened_df,
        fkey_col_to_pkey_table={
            "user": "users",
            "friend": "users",
        },
        pkey_col="id",
    )

    db.table_dict["events"] = Table(
        df=event_df,
        fkey_col_to_pkey_table={
            "user_id": "users"
        },
        pkey_col="event_id"
    )


class EventDataset(RDBenchEventDataset):
    """Event Dataset with preprocessing."""

    def make_db(self) -> Database:
        """Create the Database object with preprocessing."""

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
        preprocess_event_database(db)
        self.validate_and_correct_db(db)
        return db


def _register_event_tasks():
    from .database_factory import DatabaseFactory
    from relbench.tasks import event

    def _load_event_dataset(cache_dir: Optional[str] = None) -> Dataset:
        """Load the Event dataset."""
        cache_root_dir = os.path.join("~", ".cache", "relbench", "rel-event")
        cache_root_dir = os.path.expanduser(cache_root_dir)
        cache_dir = cache_dir if cache_dir else cache_root_dir
        return EventDataset(cache_dir=cache_dir)

    DatabaseFactory.register_dataset("event", _load_event_dataset, None)

    # Register tasks
    DatabaseFactory.register_task("event", "user-repeat", event.UserRepeatTask)
    DatabaseFactory.register_task("event", "user-ignore", event.UserIgnoreTask)
    DatabaseFactory.register_task(
        "event", "user-attendance", event.UserAttendanceTask)


_register_event_tasks()
