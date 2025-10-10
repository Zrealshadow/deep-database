import os
import warnings
from typing import Optional, Callable, Dict, Any
import numpy as np


from relbench.base import Database, Dataset, BaseTask
from relbench.datasets import get_dataset
from relbench.tasks import event, avito, trial, f1, amazon, stack


class DatabaseFactory(object):
    """
    A factory class for creating databases using a registration pattern.

    Datasets and tasks can be registered using the class methods:
    - register_dataset(db_name, loader_func, preprocessor_func=None)
    - register_task(db_name, task_name, task_class)
    """

    # Global registries
    _dataset_registry: Dict[str, Dict[str, Any]] = {}
    _task_registry: Dict[str, Dict[str, type]] = {}

    TEXT_COMPRESS_COLNAME = "text_compress"
    # a column to store the compressed text embeddings

    @classmethod
    def register_dataset(
        cls,
        db_name: str,
        loader_func: Callable,
        preprocessor_func: Optional[Callable] = None
    ) -> None:
        """
        Register a dataset loader and optional preprocessor.

        :param db_name: The name of the database.
        :param loader_func: A callable that takes (cache_dir, path) and returns a Dataset.
        :param preprocessor_func: Optional callable that takes (db: Database) and preprocesses it.
        """
        cls._dataset_registry[db_name] = {
            "loader": loader_func,
            "preprocessor": preprocessor_func
        }

    @classmethod
    def register_task(cls, db_name: str, task_name: str, task_class: type) -> None:
        """
        Register a task class for a specific database.

        :param db_name: The name of the database.
        :param task_name: The name of the task.
        :param task_class: The task class (subclass of BaseTask).
        """
        if db_name not in cls._task_registry:
            cls._task_registry[db_name] = {}
        cls._task_registry[db_name][task_name] = task_class
    
    @classmethod
    def get_registered_databases(cls) -> list:
        """Get list of all registered database names."""
        return list(cls._dataset_registry.keys())

    @classmethod
    def get_registered_tasks(cls, db_name: str) -> list:
        """Get list of all registered tasks for a database."""
        return list(cls._task_registry.get(db_name, {}).keys())

    @classmethod
    def get_db(
        cls,
        db_name: str,
        cache_dir: Optional[str] = None,
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
        if db_name not in cls._dataset_registry:
            raise ValueError(
                f"Database {db_name} not found. "
                f"Available databases: {cls.get_registered_databases()}"
            )

        dataset = cls.get_dataset(
            db_name=db_name,
            cache_dir=cache_dir,
        )

        db = dataset.get_db(upto_test_timestamp=upto_test_timestamp)

        # Apply dataset-specific preprocessor if registered
        dataset_config = cls._dataset_registry[db_name]
        if dataset_config["preprocessor"] is not None:
            dataset_config["preprocessor"](db)

        if with_text_compress:
            for _, table in db.table_dict.items():
                table.df[cls.TEXT_COMPRESS_COLNAME] = np.nan

        return db

    @classmethod
    def get_dataset(
        cls,
        db_name: str,
        cache_dir: Optional[str] = None,
    ) -> Dataset:
        """Get the dataset by name.
        :param db_name: The name of the database.
        :param cache_dir: The directory to cache the dataset.
        """
        if db_name not in cls._dataset_registry:
            raise ValueError(
                f"Unknown database name: {db_name}. "
                f"Available databases: {cls.get_registered_databases()}"
            )

        loader_func = cls._dataset_registry[db_name]["loader"]
        return loader_func(cache_dir=cache_dir)

    @classmethod
    def get_task(
            cls,
            db_name: str,
            task_name: str,
            dataset: Dataset,
    ) -> BaseTask:
        """Get a task by database name and task name.
        :param db_name: The name of the database.
        :param task_name: The name of the task.
        :param dataset: The dataset object.
        :return: The task object.
        """
        if db_name not in cls._task_registry:
            raise ValueError(
                f"Unknown database name: {db_name}. "
                f"Available databases: {list(cls._task_registry.keys())}"
            )

        if task_name not in cls._task_registry[db_name]:
            available_tasks = cls.get_registered_tasks(db_name)
            raise ValueError(
                f"Unknown task name: {task_name} for {db_name} dataset. "
                f"Available tasks: {available_tasks}"
            )

        task_class = cls._task_registry[db_name][task_name]
        cache_dir = os.path.join(dataset.cache_dir, "tasks", task_name)
        return task_class(dataset=dataset, cache_dir=cache_dir)


# ============================================================================
# Register default datasets from relbench
# ============================================================================

def _load_event_dataset(cache_dir: Optional[str] = None) -> Dataset:
    """Load the Event dataset."""
    return get_dataset("rel-event", download=True)


def _preprocess_event_database(db: Database) -> None:
    """Preprocess the Event database."""
    from .event_dataset import preprocess_event_database
    preprocess_event_database(db)


def _load_avito_dataset(cache_dir: Optional[str] = None) -> Dataset:
    """Load the Avito dataset."""
    return get_dataset("rel-avito", download=True)


def _preprocess_avito_database(db: Database) -> None:
    """Preprocess the Avito database."""
    # Reindex tables that are not properly indexed
    for _, table in db.table_dict.items():
        n = len(table.df)
        max_index = table.df.index.max()
        if n != max_index + 1:
            warnings.warn(f"Reindex table: {table}")
            table.df.reset_index(drop=True, inplace=True)


def _load_trial_dataset(cache_dir: Optional[str] = None) -> Dataset:
    """Load the Trial dataset."""
    return get_dataset("rel-trial", download=True)


def _load_f1_dataset(cache_dir: Optional[str] = None) -> Dataset:
    """Load the F1 dataset."""
    return get_dataset("rel-f1", download=True)


def _load_amazon_dataset(cache_dir: Optional[str] = None) -> Dataset:
    """Load the Amazon dataset."""
    return get_dataset("rel-amazon", download=True)


# Register datasets
DatabaseFactory.register_dataset("event", _load_event_dataset, _preprocess_event_database)
DatabaseFactory.register_dataset("avito", _load_avito_dataset, _preprocess_avito_database)
DatabaseFactory.register_dataset("trial", _load_trial_dataset)
DatabaseFactory.register_dataset("f1", _load_f1_dataset)
DatabaseFactory.register_dataset("amazon", _load_amazon_dataset)

# Register tasks
# Event tasks
DatabaseFactory.register_task("event", "user-repeat", event.UserRepeatTask)
DatabaseFactory.register_task("event", "user-ignore", event.UserIgnoreTask)
DatabaseFactory.register_task("event", "user-attendance", event.UserAttendanceTask)

# Avito tasks
DatabaseFactory.register_task("avito", "user-clicks", avito.UserClicksTask)
DatabaseFactory.register_task("avito", "ad-ctr", avito.AdCTRTask)
DatabaseFactory.register_task("avito", "user-visits", avito.UserVisitsTask)

# Trial tasks
DatabaseFactory.register_task("trial", "study-outcome", trial.StudyOutcomeTask)
DatabaseFactory.register_task("trial", "site-success", trial.SiteSuccessTask)
DatabaseFactory.register_task("trial", "study-adverse", trial.StudyAdverseTask)

# F1 tasks
DatabaseFactory.register_task("f1", "driver-dnf", f1.DriverDNFTask)
DatabaseFactory.register_task("f1", "driver-top3", f1.DriverTop3Task)


# ============================================================================
# Import custom dataset modules to trigger their self-registration
# ============================================================================
# When each module is imported, it will automatically register itself
# with the DatabaseFactory using the registration pattern.
from . import stack_dataset  # noqa: F401, E402
from . import ratebeer_dataset