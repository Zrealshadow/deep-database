import os
import pooch
import pandas as pd
import numpy as np
from dataclasses import dataclass


from relbench.base import Database, Dataset, Table, BaseTask
from relbench.utils import clean_datetime, unzip_processor
from relbench.utils import decompress_gz_file
from relbench.tasks import get_task
from relbench.datasets import get_dataset

import torch_frame
from torch_frame import stype
from torch_frame.config import (
    ImageEmbedderConfig,
    TextEmbedderConfig,
    TextTokenizerConfig,
)


from utils.util import load_col_types, save_col_types

from typing import Tuple, Dict, Any, Union, Optional


class DatabaseFactory(object):
    """
    A factory class for creating databases.
    """
    DBList = [
        "event",
        "stack",
        "avito",
        "trial",
    ]

    TEXT_COMPRESS_COLNAME = "text_compres"

    @staticmethod
    def get_db(
            db_name: str,
            cache_dir: str = None,
            with_text_compress: bool = False) -> Database:
        """
        Get a database by name.
        :param db_name: The name of the database.
        :return: The database object.
        """
        assert db_name in DatabaseFactory.DBList, f"Database {db_name} not found."

        dataset = None
        if db_name == "event":
            dataset = get_dataset("rel-event", download=True)
        elif db_name == "stack":
            dataset = StackDataset(cache_dir=cache_dir)
        elif db_name == "avito":
            dataset = get_dataset("rel-avito", download=True)
        elif db_name == "trial":
            dataset = get_dataset("rel-trial", download=True)
        else:
            raise ValueError(f"Unknown database name: {db_name}")

        db = dataset.get_db()
        if db_name == "event":
            preprocess_event_database(db)

        if with_text_compress:
            for _, table in db.table_dict.items():
                table.df[DatabaseFactory.TEXT_COMPRESS_COLNAME] = np.nan

        return db

    @staticmethod
    def get_task(
            db_name: str,
            task_name: str
    ) -> BaseTask:
        if db_name == "event":
            db_name = "rel-event"
        elif db_name == "stack":
            db_name = "rel-stack"
        elif db_name == "avito":
            db_name = "rel-avito"
        elif db_name == "trial":
            db_name = "rel-trial"
        else:
            raise ValueError(f"Unknown database name: {db_name}")
        task = get_task(db_name, task_name)
        return task


@dataclass
class TableData(object):

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    col_to_stype: Dict[str, stype]
    target_col: str

    def __post_init__(self):
        self.is_materialize = False

    def materilize(
        self,
        col_to_sep: Optional[dict[str,  Optional[str]]] = None,
        col_to_text_embedder_cfg: Optional[dict[str,
                                                TextEmbedderConfig]] = None,
        col_to_text_tokenizer_cfg: Optional[dict[str,
                                                 TextTokenizerConfig]] = None,
        col_to_image_embedder_cfg: Optional[dict[str,
                                                 ImageEmbedderConfig]] = None,
        col_to_time_format: Optional[dict[str, Optional[str]]] = None,
    ):
        if self.is_materialize:
            return

        self.train_dataset = torch_frame.data.Dataset(
            df=self.train_df,
            col_to_stype=self.col_to_stype,
            target_col=self.target_col,
            col_to_sep=col_to_sep,
            col_to_text_embedder_cfg=col_to_text_embedder_cfg,
            col_to_text_tokenizer_cfg=col_to_text_tokenizer_cfg,
            col_to_image_embedder_cfg=col_to_image_embedder_cfg,
            col_to_time_format=col_to_time_format,
        ).materialize()

        self._val_tf = self.train_dataset.convert_to_tensor_frame(self.val_df)
        self._test_tf = self.train_dataset.convert_to_tensor_frame(
            self.test_df)

        self.is_materialize = True

    def save_to_dir(
        self,
        dir_path: str
    ):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        train_df_path = os.path.join(dir_path, "train.csv")
        val_df_path = os.path.join(dir_path, "val.csv")
        test_df_path = os.path.join(dir_path, "test.csv")
        self.train_df.to_csv(train_df_path, index=False)
        self.val_df.to_csv(val_df_path, index=False)
        self.test_df.to_csv(test_df_path, index=False)
        save_col_types(dir_path, self.col_to_stype)
        with open(os.path.join(dir_path, "target_col.txt"), "w") as f:
            f.write(self.target_col)

    @staticmethod
    def load_from_dir(
        dir_path: str,
    ):

        train_df_path = os.path.join(dir_path, "train.csv")
        val_df_path = os.path.join(dir_path, "val.csv")
        test_df_path = os.path.join(dir_path, "test.csv")
        train_df = pd.read_csv(train_df_path, index_col=False)
        val_df = pd.read_csv(val_df_path, index_col=False)
        test_df = pd.read_csv(test_df_path, index_col=False)
        col_to_stype = load_col_types(dir_path)
        with open(os.path.join(dir_path, "target_col.txt"), "r") as f:
            target_col = f.read().strip()

        return TableData(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            col_to_stype=col_to_stype,
            target_col=target_col
        )

    @property
    def train_tf(self):
        if not self.is_materialize:
            self.materilize()
        return self.train_dataset.tensor_frame

    @property
    def val_tf(self):
        if not self.is_materialize:
            self.materilize()
        return self._val_tf

    @property
    def test_tf(self):
        if not self.is_materialize:
            self.materilize()
        return self._test_tf


class StackDataset(Dataset):
    """
    For stack dataset, there is an augmentation.
    We split the tags in "posts" table as a new table "tags".
    And add a new relationship table post-tags to represent the many-to-many relationship
    """

    # 3 months gap
    val_timestamp = pd.Timestamp("2020-10-01")
    test_timestamp = pd.Timestamp("2021-01-01")

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""
        url = "https://relbench.stanford.edu/data/relbench-forum-raw.zip"
        path = pooch.retrieve(
            url,
            known_hash="ad3bf96f35146d50ef48fa198921685936c49b95c6b67a8a47de53e90036745f",
            progressbar=True,
            processor=unzip_processor,
        )
        path = os.path.join(path, "raw")
        users = pd.read_csv(os.path.join(path, "Users.csv"))
        comments = pd.read_csv(os.path.join(path, "Comments.csv"))
        posts = pd.read_csv(os.path.join(path, "Posts.csv"))
        votes = pd.read_csv(os.path.join(path, "Votes.csv"))
        postLinks = pd.read_csv(os.path.join(path, "PostLinks.csv"))
        badges = pd.read_csv(os.path.join(path, "Badges.csv"))
        postHistory = pd.read_csv(os.path.join(path, "PostHistory.csv"))

        # tags = pd.read_csv(os.path.join(path, "Tags.csv")) we remove tag table here since after removing time leakage columns, all information are kept in the posts tags columns

        # remove time leakage columns
        users.drop(
            columns=["Reputation", "Views", "UpVotes",
                     "DownVotes", "LastAccessDate"],
            inplace=True,
        )

        posts.drop(
            columns=[
                "ViewCount",
                "AnswerCount",
                "CommentCount",
                "FavoriteCount",
                "CommunityOwnedDate",
                "ClosedDate",
                "LastEditDate",
                "LastActivityDate",
                # "Score",
                "LastEditorDisplayName",
                "LastEditorUserId",
            ],
            inplace=True,
        )

        # comments.drop(columns=["Score"], inplace=True)
        votes.drop(columns=["BountyAmount"], inplace=True)

        comments = clean_datetime(comments, "CreationDate")
        badges = clean_datetime(badges, "Date")
        postLinks = clean_datetime(postLinks, "CreationDate")
        postHistory = clean_datetime(postHistory, "CreationDate")
        votes = clean_datetime(votes, "CreationDate")
        users = clean_datetime(users, "CreationDate")
        posts = clean_datetime(posts, "CreationDate")

        # add an additional table "tags"
        # add an additional relationship table "post-tags"

        posts['TagList'] = posts['Tags'].str.findall(r'<(.*?)>')
        # str-> list  <bayesian><prior><elicitation> -> ['bayesian', 'prior', 'elicitation']
        post_tag = posts[['Id', 'TagList']].explode(
            'TagList').rename(columns={'TagList': 'TagName'})
        post_tag = post_tag.dropna(subset=['TagName']).reset_index(drop=True)

        tags = pd.DataFrame(post_tag['TagName'].unique(), columns=['TagName'])
        tags['TagId'] = range(1, len(tags) + 1)
        post_tag = post_tag.merge(tags, on='TagName', how='left')[
            ['Id', 'TagId']]

        # clear the schema name
        post_tag['PostId'] = post_tag['Id']
        post_tag['Id'] = range(1, len(post_tag) + 1)
        tags['Id'] = tags['TagId']
        tags.drop(columns=['TagId'], inplace=True)

        # drop 'Tags' column in posts
        posts.drop(columns=['Tags'], inplace=True)
        posts.drop(columns=['TagList'], inplace=True)

        tables = {}

        tables["comments"] = Table(
            df=pd.DataFrame(comments),
            fkey_col_to_pkey_table={
                "UserId": "users",
                "PostId": "posts",
            },
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["badges"] = Table(
            df=pd.DataFrame(badges),
            fkey_col_to_pkey_table={
                "UserId": "users",
            },
            pkey_col="Id",
            time_col="Date",
        )

        tables["postLinks"] = Table(
            df=pd.DataFrame(postLinks),
            fkey_col_to_pkey_table={
                "PostId": "posts",
                "RelatedPostId": "posts",  # is this allowed? two foreign keys into the same primary
            },
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["postHistory"] = Table(
            df=pd.DataFrame(postHistory),
            fkey_col_to_pkey_table={"PostId": "posts", "UserId": "users"},
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["votes"] = Table(
            df=pd.DataFrame(votes),
            fkey_col_to_pkey_table={"PostId": "posts", "UserId": "users"},
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["users"] = Table(
            df=pd.DataFrame(users),
            fkey_col_to_pkey_table={},
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["posts"] = Table(
            df=pd.DataFrame(posts),
            fkey_col_to_pkey_table={
                "OwnerUserId": "users",
                "ParentId": "posts",  # notice the self-reference
                "AcceptedAnswerId": "posts",
            },
            pkey_col="Id",
            time_col="CreationDate",
        )

        # add the new tables
        tables["tags"] = Table(
            df=pd.DataFrame(tags),
            fkey_col_to_pkey_table={},
            pkey_col="Id",
            time_col=None,
        )

        # add the new relationship table
        tables["postTag"] = Table(
            df=pd.DataFrame(post_tag),
            fkey_col_to_pkey_table={
                "PostId": "posts",
                "TagId": "tags",
            },
            pkey_col="Id",
            time_col=None,
        )

        return Database(tables)


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
    event_df["event_id"].replace(event_id2index, inplace=True)
    # map the event_id in event_interest and event_attendees
    event_interest_df["event"].replace(event_id2index, inplace=True)
    event_attendees_flattened_df["event"].replace(event_id2index, inplace=True)

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
