"""
Task extensions - Add custom methods to both your tasks and library tasks.

This module uses monkey patching to add shared methods to task classes,
including library tasks that cannot be modified directly.

Usage:
    from utils.task import task_extensions

    # Now all tasks (library and custom) have the extended methods
    task.mask_features(db)
"""

from relbench.base import BaseTask, Database


def default_mask_features(self, db: Database) -> None:
    """
    Mask features in the database that may contain truth values.

    The database may contain the truth values in feature columns (data leakage).
    This method masks those task-specific features and drops them for modeling.

    Default implementation does nothing. Override in task subclasses to implement
    task-specific feature masking logic.

    Args:
        db: Database object containing tables to mask

    Example:
        class ActiveUserPredictionTask(EntityTask):
            def mask_features(self, db: Database) -> None:
                # Drop columns that leak target information
                if 'future_activity' in db.table_dict['users'].df.columns:
                    db.table_dict['users'].df.drop('future_activity', axis=1, inplace=True)
    """
    pass

# ============================================================================
# Apply monkey patches to add methods to all task classes
# ============================================================================

def extend_task_classes():
    """
    Add custom methods to all task classes (both custom and library).

    This function monkey patches the BaseTask class to add shared functionality
    to all tasks, including library tasks that cannot be modified directly.

    Methods added:
        - mask_database(db): Mask features that may leak target information
    """
    BaseTask.mask_database = default_mask_features
    print("✓ Task extensions applied: BaseTask.mask_features")


# Auto-apply extensions when this module is imported
extend_task_classes()
