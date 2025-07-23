import os
import pooch
import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

from relbench.base import Database, Dataset, Table, BaseTask, TaskType
from relbench.utils import clean_datetime, unzip_processor
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
from utils.task import ActiveUserPredictionTask, BeerPositiveRatePredictionTask, PlacePositivePredictionTask
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
        "ratebeer"
    ]

    TEXT_COMPRESS_COLNAME = "text_compress"

    @staticmethod
    def get_db(
            db_name: str,
            cache_dir: Optional[str] = None,
            path: Optional[str] = None,
            with_text_compress: bool = False) -> Database:
        """
        Get a database by name.
        :param db_name: The name of the database.
        :return: The database object.
        """
        assert db_name in DatabaseFactory.DBList, f"Database {db_name} not found."

        dataset = DatabaseFactory.get_dataset(
            db_name=db_name,
            cache_dir=cache_dir,
            path=path,
        )

        db = dataset.get_db()
        if db_name == "event":
            preprocess_event_database(db)
        elif db_name == "avito":
            # case by case:
            # there are some tables in this dataset are not reindex
            for _, table in db.table_dict.items():
                n = len(table.df)
                max_index = table.df.index.max()
                if n != max_index + 1:
                    table.df.reset_index(drop=True, inplace=True)

        if with_text_compress:
            for _, table in db.table_dict.items():
                table.df[DatabaseFactory.TEXT_COMPRESS_COLNAME] = np.nan

        return db

    @staticmethod
    def get_dataset(
        db_name: str,
        cache_dir: Optional[str] = None,
        path: Optional[str] = None,
    ) -> Dataset:
        dataset = None
        if db_name == "event":
            dataset = get_dataset("rel-event", download=True)
        elif db_name == "stack":
            dataset = StackDataset(cache_dir=cache_dir)
        elif db_name == "avito":
            dataset = get_dataset("rel-avito", download=True)
        elif db_name == "trial":
            dataset = get_dataset("rel-trial", download=True)
        elif db_name == "ratebeer":
            # if path is None:
            #     raise ValueError(
            #         "Local Path must be provided for RateBeer dataset.")
            dataset = RateBeerDataset(
                path,
                cache_dir=cache_dir
            )
        else:
            raise ValueError(f"Unknown database name: {db_name}")
        return dataset

    @staticmethod
    def get_task(
            db_name: str,
            task_name: str,
            dataset: Optional[Dataset] = None,
    ) -> BaseTask:
        if db_name == "event":
            db_name = "rel-event"
        elif db_name == "stack":
            db_name = "rel-stack"
        elif db_name == "avito":
            db_name = "rel-avito"
        elif db_name == "trial":
            db_name = "rel-trial"
        elif db_name == "ratebeer":
            if dataset is None:
                raise ValueError(
                    "Dataset must be provided for RateBeer dataset.")
            # assign specific task
            cache_dir = os.path.join(dataset.cache_dir, "tasks", task_name)
            if task_name == "user-active":
                task = ActiveUserPredictionTask(
                    dataset=dataset,
                    cache_dir=cache_dir,
                )
                return task
            elif task_name == "beer-positive":
                task = BeerPositiveRatePredictionTask(
                    dataset=dataset,
                    cache_dir=cache_dir,
                )
                return task
            elif task_name == "place-positive":
                task = PlacePositivePredictionTask(
                    dataset=dataset,
                    cache_dir=cache_dir,
                )
                return task
            else:
                raise ValueError(
                    f"Unknown task name: {task_name} for RateBeer dataset.")

        else:
            raise ValueError(f"Unknown database name: {db_name}")
        task = get_task(db_name, task_name)
        return task


TextEmbedderCFG = Optional[TextEmbedderConfig]
TextTokenizerCFG = Union[dict[str, TextTokenizerConfig],
                         TextTokenizerConfig, None]
ImageEmbedderCFG = Union[dict[str, ImageEmbedderConfig],
                         ImageEmbedderConfig, None]


@dataclass
class TableData(object):

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    col_to_stype: Dict[str, stype]
    target_col: str
    task_type: TaskType

    def __post_init__(self):
        self.is_materialize = False

    def materilize(
        self,
        col_to_sep: Optional[dict[str,  Optional[str]]] = None,
        col_to_text_embedder_cfg: Optional[TextEmbedderConfig] = None,
        col_to_text_tokenizer_cfg: TextEmbedderCFG = None,
        col_to_image_embedder_cfg: TextTokenizerCFG = None,
        col_to_time_format: ImageEmbedderCFG = None,
    ):
        if self.is_materialize:
            return

        train_dataset = torch_frame.data.Dataset(
            df=self.train_df,
            col_to_stype=self.col_to_stype,
            target_col=self.target_col,
            col_to_sep=col_to_sep,
            col_to_text_embedder_cfg=col_to_text_embedder_cfg,
            col_to_text_tokenizer_cfg=col_to_text_tokenizer_cfg,
            col_to_image_embedder_cfg=col_to_image_embedder_cfg,
            col_to_time_format=col_to_time_format,
        ).materialize()

        self._train_tf = train_dataset.tensor_frame
        self._col_stats = train_dataset.col_stats
        self._val_tf = train_dataset.convert_to_tensor_frame(
            self.val_df)
        self._test_tf = train_dataset.convert_to_tensor_frame(
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
            f.write(self.target_col+"\n")
            f.write(self.task_type.name+"\n")

        # check if is materialize
        if self.is_materialize:
            # save the tensorframe
            train_tf_path = os.path.join(dir_path, "train_tf.pt")
            val_tf_path = os.path.join(dir_path, "val_tf.pt")
            test_tf_path = os.path.join(dir_path, "test_tf.pt")
            torch_frame.save(self.train_tf, self.col_stats, train_tf_path)
            torch_frame.save(self.val_tf, None, path=val_tf_path)
            torch_frame.save(self.test_tf, None, path=test_tf_path)

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
            target_col = f.readline().strip()
            task_type = TaskType[f.readline().strip()]

        table_data = TableData(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            col_to_stype=col_to_stype,
            target_col=target_col,
            task_type=task_type,
        )

        # check if there is train_tf.pt or others
        train_tf_path = os.path.join(dir_path, "train_tf.pt")
        val_tf_path = os.path.join(dir_path, "val_tf.pt")
        test_tf_path = os.path.join(dir_path, "test_tf.pt")
        if os.path.exists(train_tf_path):
            assert os.path.exists(val_tf_path)
            assert os.path.exists(test_tf_path)

            table_data.is_materialize = True
            # update the train_tf, val_tf, test_tf, col_stats
            table_data._train_tf, table_data._col_stats = torch_frame.load(
                path=train_tf_path)
            table_data._val_tf, _ = torch_frame.load(path=val_tf_path)
            table_data._test_tf, _ = torch_frame.load(path=test_tf_path)
            print(f" ==> load material dataset from {dir_path}")
        else:
            table_data.is_materialize = False
            print(
                f" ==> load raw dataset from {dir_path}, need material first")
        return table_data

    @property
    def train_tf(self):
        if not self.is_materialize:
            raise ValueError(
                "The tensor frame is not materialized. Please call materilize() first."
            )
        return self._train_tf

    @property
    def val_tf(self):
        if not self.is_materialize:
            raise ValueError(
                "The tensor frame is not materialized. Please call materilize() first."
            )
        return self._val_tf

    @property
    def test_tf(self):
        if not self.is_materialize:
            raise ValueError(
                "The tensor frame is not materialized. Please call materilize() first."
            )
        return self._test_tf

    @property
    def col_stats(self):
        if not self.is_materialize:
            raise ValueError(
                "The tensor frame is not materialized. Please call materilize() first."
            )
        return self._col_stats

    @property
    def col_names_dict(self):
        if not self.is_materialize:
            raise ValueError(
                "The tensor frame is not materialized. Please call materilize() first."
            )
        return self.train_tf.col_names_dict


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


class RateBeerDataset(Dataset):
    val_timestamp = pd.Timestamp("2023-06-01")
    test_timestamp = pd.Timestamp("2024-06-01")

    DB_URL = "https://www.dropbox.com/scl/fi/exwygxep7vdvq55uiq28r/db.zip?rlkey=o7q0r8nw758p4wxx1wka9ubuj&st=rg3gvkxg&dl=1"

    def __init__(self, path: str, cache_dir: Optional[str] = None):
        super().__init__(cache_dir=cache_dir)
        self.dir_path = path

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""
        # Create cache directory if it doesn't exist
        # os.makedirs(CACHE_DIR, exist_ok=True)

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
            csv_path = os.path.join(self.dir_path, f"{table_name}.csv")
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
