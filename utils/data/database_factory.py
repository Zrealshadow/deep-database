import os
import warnings
from typing import Optional
import numpy as np


from relbench.base import Database, Dataset, BaseTask
from relbench.datasets import get_dataset
from relbench.tasks import stack, event, f1, amazon, avito, trial

from utils.task import ActiveUserPredictionTask, BeerPositiveRatePredictionTask, PlacePositivePredictionTask
from .stack_dataset import StackDataset
from .ratebeer_dataset import RateBeerDataset
from .event_dataset import preprocess_event_database


class DatabaseFactory(object):
    """
    A factory class for creating databases.
    """
    DBList = [
        "event",
        "stack",
        "avito",
        "trial",
        "ratebeer",
        "f1",
        "amazon"
    ]

    TEXT_COMPRESS_COLNAME = "text_compress"
    # a column to store the compressed text embeddings

    @staticmethod
    def get_db(
        db_name: str,
        cache_dir: Optional[str] = None,
        path: Optional[str] = None,
        with_text_compress: bool = False,
        upto_test_timestamp: bool = True,
    ) -> Database:
        """
        Get a database by name.
        :param db_name: The name of the database.
        :param cache_dir: The directory to cache the database.
        :param path: The local path to the database
        :param with_text_compress: Whether to add a column to store the compressed text embeddings.
            compressed text is the preprocessing to concatenate all columns with text type within a table.
        :param upto_test_timestamp: Whether to filter the data up to the test timestamp.
            If false, we will use the whole time data for database (including test dataset).
        :return: The database object.
        """
        assert db_name in DatabaseFactory.DBList, f"Database {db_name} not found."

        dataset = DatabaseFactory.get_dataset(
            db_name=db_name,
            cache_dir=cache_dir,
            path=path,
        )

        db = dataset.get_db(upto_test_timestamp=upto_test_timestamp)
        if db_name == "event":
            preprocess_event_database(db)
        elif db_name == "avito":
            # case by case:
            # there are some tables in this dataset are not reindex
            for _, table in db.table_dict.items():
                n = len(table.df)
                max_index = table.df.index.max()
                if n != max_index + 1:
                    warnings.warn(f"Reindex table: {table}")
                    table.df.reset_index(drop=True, inplace=True)

            pass
            # TODO: I manually reindex and cache the database in my local home folder.
            # Currently, above branch won't be executed for me.
            # however, for other user, it will stil lead to promblem

            # we need to reimplement a Wrapper for Event and Avito
            # to preprocess again this dataset above the Class from relbench

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
        """Get the dataset by name.
        :param db_name: The name of the database.
        :param cache_dir: The directory to cache the dataset.
        :param path: The local path to the dataset, used for RateBeer dataset.(
            first get the dataset need path from local path.
            After the dataset is cached, we can load it from cache directly.
        )
        """
        cache_root_dir = os.path.join("~", ".cache", "relbench")
        cache_root_dir = os.path.expanduser(cache_root_dir)
        dataset = None
        if db_name == "event":
            dataset = get_dataset("rel-event", download=True)
        elif db_name == "stack":
            cache_dir = cache_dir if cache_dir else \
                os.path.join(cache_root_dir, "stack")
            print("Stack dataset cache dir:", cache_dir)
            dataset = StackDataset(cache_dir=cache_dir)
        elif db_name == "avito":
            dataset = get_dataset("rel-avito", download=True)
        elif db_name == "trial":
            dataset = get_dataset("rel-trial", download=True)
        elif db_name == "ratebeer":
            # add default cache_dir
            cache_dir = cache_dir if cache_dir else \
                os.path.join(cache_root_dir, "ratebeer")
            dataset = RateBeerDataset(
                path,
                cache_dir=cache_dir
            )
        elif db_name == "f1":
            dataset = get_dataset("rel-f1", download=True)
        elif db_name == "amazon":
            dataset = get_dataset("rel-amazon", download=True)
        else:
            raise ValueError(f"Unknown database name: {db_name}")
        return dataset

    @staticmethod
    def get_task(
            db_name: str,
            task_name: str,
            dataset: Dataset,
    ) -> BaseTask:
        cache_dir = os.path.join(dataset.cache_dir, "tasks", task_name)
        kwargs = {
            "dataset": dataset,
            "cache_dir": cache_dir,
        }

        if db_name == "event":
            if task_name == "user-repeat":
                task_type = event.UserRepeatTask
            elif task_name == "user-ignore":
                task_type = event.UserIgnoreTask
            elif task_name == "user-attendance":
                task_type = event.UserAttendanceTask
            else:
                raise ValueError(
                    f"Unknown task name: {task_name} for Event dataset.")
        elif db_name == "stack":
            if task_name == "user-engagement":
                task_type = stack.UserEngagementTask
            elif task_name == "user-badge":
                task_type = stack.UserBadgeTask
            elif task_name == "post-vote":
                task_type = stack.PostVotesTask
            else:
                raise ValueError(
                    f"Unknown task name: {task_name} for Stack dataset.")
        elif db_name == "avito":
            if task_name == "user-clicks":
                task_type = avito.UserClicksTask
            elif task_name == "ad-ctr":
                task_type = avito.AdCTRTask
            elif task_name == "user-visits":
                task_type = avito.UserVisitsTask
            else:
                raise ValueError(
                    f"Unknown task name: {task_name} for Avito dataset.")
        elif db_name == "trial":
            if task_name == "study-outcome":
                task_type = trial.StudyOutcomeTask
            elif task_name == "site-success":
                task_type = trial.SiteSuccessTask
            elif task_name == "study-adverse":
                task_type = trial.StudyAdverseTask
            else:
                raise ValueError(
                    f"Unknown task name: {task_name} for Trial dataset.")
        elif db_name == "f1":
            if task_name == "driver-dnf":
                task_type = f1.DriverDNFTask
            elif task_name == "driver-top3":
                task_type = f1.DriverTop3Task
            else:
                raise ValueError(
                    f"Unknown task name: {task_name} for F1 dataset.")
        elif db_name == "amazon":
            pass
            # TODO:
        elif db_name == "ratebeer":
            cache_dir = os.path.join(dataset.cache_dir, "tasks", task_name)
            if task_name == "user-active":
                task_type = ActiveUserPredictionTask
            elif task_name == "beer-positive":
                task_type = BeerPositiveRatePredictionTask
            elif task_name == "place-positive":
                task_type = PlacePositivePredictionTask
            else:
                raise ValueError(
                    f"Unknown task name: {task_name} for RateBeer dataset.")

        else:
            raise ValueError(f"Unknown database name: {db_name}")

        return task_type(**kwargs)
