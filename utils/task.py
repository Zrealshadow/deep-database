import duckdb
import pandas as pd

from relbench.base import Database, RecommendationTask, Table, TaskType, EntityTask, Dataset
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
    mae,
    r2,
    rmse,
    roc_auc,
)


class ActiveUserPredictionTask(EntityTask):
    """
    Entity Prediction Task,
    Predict whether a user will be active in the next season (90 days)
    "active" -> user has at least 5 beer ratings.
    """

    task_type = TaskType.BINARY_CLASSIFICATION
    time_col = "timestamp"
    target_col = "is_active"
    timedelta = pd.Timedelta(days=90)
    metrics = [roc_auc, accuracy, f1, average_precision]
    num_eval_timestamps = 1
    entity_table = "users"
    entity_col = "user_id"

    def make_table(
        self,
        db: Database,
        timestamps: pd.Series,
    ) -> Table:

        if timestamps.empty:
            raise ValueError("Timestamps cannot be empty.")

        timestamp_df = pd.DataFrame({
            self.time_col: timestamps,
        })
        user_df = db.table_dict['users'].df
        beer_rating_df = db.table_dict['beer_ratings'].df

        con = duckdb.connect()
        con.register("t", timestamp_df)
        con.register("user_df", user_df)
        con.register("beer_rating_df", beer_rating_df)

        sql = f"""
                SELECT 
            rating_info.user_id,
            t.timestamp,
            CASE 
                WHEN COALESCE(COUNT(DISTINCT rating_id), 0) > 10 THEN 1
                ELSE 0
            END AS is_active,
        FROM
            timestamp_df t
        LEFT JOIN
        (   
            SELECT 
                user_df.user_id AS user_id,
                beer_rating_df.created_at AS ActionDate,
                beer_rating_df.rating_id AS rating_id
            From
                user_df
            LEFT JOIN
                beer_rating_df
            ON 
             user_df.user_id = beer_rating_df.user_id
             
        ) AS rating_info
        ON
            rating_info.ActionDate > t.timestamp AND
            rating_info.ActionDate <= t.timestamp + INTERVAL '{self.timedelta} days'
        GROUP BY
            t.timestamp,
            rating_info.user_id
        """
        df = con.execute(sql).fetchdf()
        con.close()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col='timestamp'
        )

    def _get_table(self, split: str) -> Table:
        """
        reimplement the _get_table method in BaseTask
        handle the train set, which only consider the last two years from val_timestamp
        """
        db = self.dataset.get_db(upto_test_timestamp=split != "test")
        timestamps = ratebeer_timestamp_decision(
            split=split,
            dataset=self.dataset,
            db=db,
            timedelta=self.timedelta
        )
        table = self.make_table(db, timestamps)
        table = self.filter_dangling_entities(table)
        return table


class PlacePositivePredictionTask(EntityTask):
    r"""
    Predict whether a place will receive positive rating in the next half of the year (180 days).
    average_rating > 75 is considered positive
    """
    task_type = TaskType.BINARY_CLASSIFICATION
    time_col = "timestamp"
    target_col = "is_positive"
    timedelta = pd.Timedelta(days=180)
    metrics = [roc_auc, accuracy, f1, average_precision]
    num_eval_timestamps = 1
    entity_table = "places"
    entity_col = "place_id"

    def make_table(
        self,
        db: Database,
        timestamps: pd.Series,
    ) -> Table:
        place_df = db.table_dict['places'].df
        place_rating_df = db.table_dict['place_ratings'].df
        timestamp_df = pd.DataFrame({
            self.time_col: timestamps,
        })

        con = duckdb.connect()
        con.register("place_df", place_df)
        con.register("place_rating_df", place_rating_df)
        con.register("t", timestamp_df)

        sql = f"""
        SELECT 
            rating_info.place_id,
            t.timestamp,
            CASE 
                WHEN AVG(rating_info.score) > 75 THEN 1
                ELSE 0
            END AS is_positive
        FROM
            timestamp_df t
        LEFT JOIN
        (   
            SELECT 
                place_df.place_id AS place_id,
                place_rating_df.created_at AS ActionDate,
                place_rating_df.rating_id AS rating_id,
                place_rating_df.total_score AS score,
            From
                place_df
            LEFT JOIN
                place_rating_df
            ON 
                place_df.place_id = place_rating_df.place_id
             
        ) AS rating_info
        ON
            rating_info.ActionDate > t.timestamp AND
            rating_info.ActionDate <= t.timestamp + INTERVAL '{self.timedelta} days'
        GROUP BY
            t.timestamp,
            rating_info.place_id;
        """

        df = con.execute(sql).fetchdf()
        con.close()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col
        )

    def _get_table(self, split: str) -> Table:
        """
        reimplement the _get_table method in BaseTask
        handle the train set, which only consider the last two years from val_timestamp
        """
        db = self.dataset.get_db(upto_test_timestamp=split != "test")
        timestamps = ratebeer_timestamp_decision(
            split=split,
            dataset=self.dataset,
            db=db,
            timedelta=self.timedelta
        )
        table = self.make_table(db, timestamps)
        table = self.filter_dangling_entities(table)
        return table


class BeerPositiveRatePredictionTask(EntityTask):
    r"""
    Predict whether a beer will be rated positively in the next half of the year (180 days).
    rating > 3.5 is considered positive, only consider the number of ratings is larger than 3.
    positive rate is the ratio of positive ratings over all ratings.
    """

    task_type = TaskType.REGRESSION
    time_col = "timestamp"
    target_col = "pos_rate"
    timedelta = pd.Timedelta(days=180)
    metrics = [r2, mae, rmse]
    num_eval_timestamps = 1
    entity_table = "beers"
    entity_col = "beer_id"

    def make_table(
        self,
        db: Database,
        timestamps: pd.Series,
    ) -> Table:
        beer_df = db.table_dict['beers'].df
        beer_rating_df = db.table_dict['beer_ratings'].df
        timestamp_df = pd.DataFrame({
            self.time_col: timestamps,
        })

        con = duckdb.connect()
        con.register("beer_df", beer_df)
        con.register("beer_rating_df", beer_rating_df)
        con.register("t", timestamp_df)

        sql = f"""
            SELECT 
                rating_info.beer_id,
                t.timestamp,
                SUM(CASE WHEN rating_info.score > 3.5 THEN 1 ELSE 0 END) * 1.0 
                    / COUNT(rating_info.rating_id) AS pos_rate
            FROM
                timestamp_df t
            LEFT JOIN
            (   
                SELECT 
                    beer_df.beer_id AS beer_id,
                    beer_rating_df.created_at AS ActionDate,
                    beer_rating_df.rating_id AS rating_id,
                    beer_rating_df.total_score AS score,
                From
                    beer_df
                LEFT JOIN
                    beer_rating_df
                ON 
                    beer_df.beer_id = beer_rating_df.beer_id
            ) AS rating_info
            ON
                rating_info.ActionDate > t.timestamp AND
                rating_info.ActionDate <= t.timestamp + INTERVAL '180 days'
            GROUP BY
                t.timestamp,
                rating_info.beer_id
            HAVING
                COUNT(rating_info.rating_id) > 3;
            """

        df = con.execute(sql).fetchdf()
        con.close()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col
        )

    def _get_table(self, split: str) -> Table:
        """
        reimplement the _get_table method in BaseTask
        handle the train set, which only consider the last two years from val_timestamp
        """
        db = self.dataset.get_db(upto_test_timestamp=split != "test")
        timestamps = ratebeer_timestamp_decision(
            split=split,
            dataset=self.dataset,
            db=db,
            timedelta=self.timedelta
        )
        table = self.make_table(db, timestamps)
        table = self.filter_dangling_entities(table)
        return table


def ratebeer_timestamp_decision(
    split: str,
    dataset: Dataset,
    db: Database,
    timedelta: pd.Timedelta
) -> pd.DatetimeIndex:
    """
    Helper function to get timestamp for train, test, val split
    """
    if split == "train":
        start = dataset.val_timestamp - timedelta
        end = dataset.val_timestamp - timedelta - pd.Timedelta(days=365)
        # only catch the recent one years data
        freq = - timedelta
    elif split == "val":
        if dataset.val_timestamp + timedelta > db.max_timestamp:
            raise RuntimeError(
                "val timestamp + timedelta is larger than max timestamp! "
                "This would cause val labels to be generated with "
                "insufficient aggregation time."
            )
        start = dataset.val_timestamp
        end = min(
            dataset.val_timestamp,
            dataset.test_timestamp - timedelta,
        )
        freq = timedelta
    elif split == "test":
        if dataset.test_timestamp + timedelta > db.max_timestamp:
            raise RuntimeError(
                "test timestamp + timedelta is larger than max timestamp! "
                "This would cause test labels to be generated with "
                "insufficient aggregation time."
            )
        start = dataset.test_timestamp
        end = min(
            dataset.test_timestamp + timedelta,
            db.max_timestamp - timedelta,
        )
        freq = timedelta
    timestamps = pd.date_range(start=start, end=end, freq=freq)
    return timestamps
