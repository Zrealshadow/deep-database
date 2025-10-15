import pandas as pd

from relbench.base import (
    EntityTask,
    TaskType,
    Database,
    Table,
)


class OlistOrderDelayTask(EntityTask):
    """
    A task to predict whether an order is delayed in delivery.
    Don't like other temporal prediction tasks, this task is more like tabular data classification.
    However, we construct it follow the temporal prediction task interface.

    train/val/test delay ratio:0.07/0.11/0.05
    """

    task_type = TaskType.BINARY_CLASSIFICATION
    time_col = "timestamp"
    target_col = "is_delayed"
    entity_table = "order"
    entity_col = "order_id"

    timedelta = pd.Timedelta(days=1)
    # timedelta is not used in this class
    # this class is not a temporal prediction task

    def make_table(
        self,
        db: Database,
        timestamps: pd.Series,
    ) -> Table:
        start_ts, end_ts = timestamps
        order_table = db.table_dict["order"]
        order_df = order_table.df.copy()

        # drop all delivered_customer_time or estimated_delivery_time is NaT
        order_df = order_df.dropna(
            subset=["delivered_customer_time", "estimated_delivery_time"])

        # calculate whether the order is delayed
        order_df[self.target_col] = (
            order_df["delivered_customer_time"] >
            order_df["estimated_delivery_time"]
        ).astype(int)

        # calculate whether the order is within the time window
        split_select = (order_df[order_table.time_col] >= start_ts) & (
            order_df[order_table.time_col] <= end_ts
        )
        order_df = order_df[split_select].reset_index(drop=True)
        order_df[self.time_col] = end_ts

        # keep only entity_col, time_col, target_col
        order_df = order_df[[self.entity_col, self.time_col, self.target_col]]

        return Table(
            df=order_df,
            fkey_col_to_pkey_table={
                self.entity_col: self.entity_table
            },
            pkey_col=None,
            time_col=self.time_col,
        )

    def _get_table(self, split: str) -> Table:
        db = self.dataset.get_db(upto_test_timestamp=split != "test")

        timestamps_dict = {
            "train": pd.Series([db.min_timestamp, self.dataset.val_timestamp]),
            "val": pd.Series([self.dataset.val_timestamp, self.dataset.test_timestamp]),
            "test": pd.Series([self.dataset.test_timestamp, db.max_timestamp]),
        }

        timestamps = timestamps_dict[split]
        table = self.make_table(db, timestamps)
        # table = self.filter_dangling_entities(table)
        return table

    def mask_database(db: Database) -> None:
        """
        ReImplement the mask_database method which is patched to BaseTask
        Mask features in the database that may contain truth values.
        """
        assert "order" in db.table_dict, "order table not in db"
        db.table_dict["order"].df.drop(
            ['delivered_customer_time', 'estimated_delivery_time'], axis=1, inplace=True
        )
