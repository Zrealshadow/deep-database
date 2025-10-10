# Adding New Datasets and Tasks to DatabaseFactory

This guide explains how to add new datasets and tasks using the registration pattern.

## Overview

The `DatabaseFactory` uses a **registration pattern** instead of if-else conditions. This means:
- ✅ Adding new datasets/tasks requires NO modification to the factory class
- ✅ Each dataset/task is self-contained and registered independently
- ✅ Easy to maintain and extend
- ✅ Two approaches: default registration (in `database_factory.py`) or self-registration (in dataset module)

## Approach 1: Default Registration (Simple Datasets)

For simple datasets from relbench or datasets that don't need custom classes, register them directly in `database_factory.py`.

### Step 1: Create loader and preprocessor functions

Add these functions in `database_factory.py` before the registration section:

```python
def _load_xxx_dataset(cache_dir: Optional[str] = None, path: Optional[str] = None) -> Dataset:
    """Load the XXXDataset dataset."""
    return get_dataset("rel-xxxdataset", download=True)

def _preprocess_xxx_database(db: Database) -> None:
    """Preprocess the XXXDataset database (optional)."""
    # Your preprocessing logic here
    pass
```

### Step 2: Register the dataset and tasks

In the "Register default datasets" section:

```python
# Import task modules at top of file
from relbench.tasks import xxxdataset

# Register dataset
DatabaseFactory.register_dataset("xxxdataset", _load_xxx_dataset, _preprocess_xxx_database)

# Register tasks
DatabaseFactory.register_task("xxxdataset", "task1", xxxdataset.Task1Class)
DatabaseFactory.register_task("xxxdataset", "task2", xxxdataset.Task2Class)
```

**That's it!** Your dataset is now available.

## Approach 2: Self-Registration (Custom Datasets)

For custom datasets with their own Dataset class, use self-registration in the dataset module itself.

### Step 1: Create your dataset file (e.g., `xxxdataset.py`)

```python
import os
from typing import Optional
from relbench.base import Database, Dataset, Table

class XXXDataset(Dataset):
    """Your custom dataset implementation."""

    def make_db(self) -> Database:
        """Process the raw files into a database."""
        # Your dataset loading logic here
        tables = {}
        # ... create tables ...
        return Database(tables)
```

### Step 2: Add registration function at the end of the file

```python
# ============================================================================
# Self-registration with DatabaseFactory
# ============================================================================

def _register_xxxdataset():
    """Register XXXDataset and tasks with DatabaseFactory."""
    from .database_factory import DatabaseFactory
    from relbench.tasks import xxxmodule  # Or your custom tasks

    def _load_xxxdataset(cache_dir: Optional[str] = None, path: Optional[str] = None) -> Dataset:
        """Load the XXXDataset."""
        cache_root_dir = os.path.join("~", ".cache", "relbench")
        cache_root_dir = os.path.expanduser(cache_root_dir)
        cache_dir = cache_dir if cache_dir else os.path.join(cache_root_dir, "xxxdataset")
        return XXXDataset(cache_dir=cache_dir)

    # Register dataset
    DatabaseFactory.register_dataset("xxxdataset", _load_xxxdataset)

    # Register tasks
    DatabaseFactory.register_task("xxxdataset", "task1", xxxmodule.Task1Class)
    DatabaseFactory.register_task("xxxdataset", "task2", xxxmodule.Task2Class)


# Auto-register when this module is imported
_register_xxxdataset()
```

### Step 3: Import the module in `database_factory.py`

At the bottom of `database_factory.py`, in the "Import custom dataset modules" section:

```python
from . import xxxdataset  # noqa: F401, E402
```

**Done!** The dataset auto-registers when imported.

## Benefits of This Pattern

1. **No modification needed**: The `DatabaseFactory` class never needs to be modified
2. **Clear separation**: Each dataset's logic is self-contained
3. **Easy to test**: Each loader/preprocessor can be tested independently
4. **Discoverable**: Use `DatabaseFactory.get_registered_databases()` and `DatabaseFactory.get_registered_tasks("db_name")` to see what's available
5. **Better error messages**: Automatically suggests available options when an unknown dataset/task is requested

## Utility Methods

```python
# Get list of all registered databases
databases = DatabaseFactory.get_registered_databases()

# Get list of all tasks for a specific database
tasks = DatabaseFactory.get_registered_tasks("event")
```

---

# Testing

## Running Tests

The test suite is located in `test/test_database_factory.py` and supports two modes:

### Fast Mode (Unit Tests Only - Default)
```bash
# Run from project root
python -m unittest test.test_database_factory

# Or run directly
python test/test_database_factory.py
```

**What it does:**
- ✅ Tests dataset/task registration
- ✅ Tests error handling
- ✅ Tests registry structure
- ⏭️ Skips integration tests (no dataset downloads)

### Integration Mode (End-to-End Tests)
```bash
# Set environment variable
XXX_RUN_INTEGRATION_TESTS=1 python -m unittest test.test_database_factory

# Or export and run
export XXX_RUN_INTEGRATION_TESTS=1
python test/test_database_factory.py
```

**What it does:**
- ⏭️ Skips unit tests
- ✅ Downloads and tests actual datasets
- ✅ Tests all registered databases and tasks
- ✅ Tests database loading with different parameters

**Note:** Integration tests download datasets and may take 5-30 minutes depending on your connection.

## Adding Tests for Your New Dataset

### Step 1: Add to Integration Test Cases

Edit `test/test_database_factory.py` and add your dataset to `DATASET_TEST_CASES`:

```python
class TestDatabaseFactoryIntegration(unittest.TestCase):
    # Test configuration: (db_name, task_name)
    DATASET_TEST_CASES = [
        ("event", "user-repeat"),
        ("avito", "user-clicks"),
        ("trial", "study-outcome"),
        ("f1", "driver-dnf"),
        ("stack", "user-engagement"),
        ("ratebeer", "user-active"),
        ("xxxdataset", "xxxtask"),  # Add your dataset here!
    ]
```

That's it! Your dataset will now be tested with all existing integration tests automatically.

### Step 2: (Optional) Add Unit Tests for Task Registration

If you want to add specific unit tests for your dataset's tasks, add to `TestDatabaseFactoryRegistration`:

```python
@unittest.skipIf(XXX_RUN_INTEGRATION_TESTS, "Skipping unit tests when integration tests are enabled")
class TestDatabaseFactoryRegistration(unittest.TestCase):
    # ... existing tests ...

    def test_3_xxxdataset_tasks_registered(self):
        """Test that xxxdataset tasks are registered."""
        xxxdataset_tasks = DatabaseFactory.get_registered_tasks("xxxdataset")
        self.assertIn("task1", xxxdataset_tasks)
        self.assertIn("task2", xxxdataset_tasks)
        print(f"✅ XXXDataset tasks: {xxxdataset_tasks}")
```

**Naming convention:** Use `test_3_` prefix to group with other task registration tests (they run alphabetically).

### Step 3: Run Your Tests

```bash
# Test registration only (fast)
python -m unittest test.test_database_factory.TestDatabaseFactoryRegistration.test_3_xxxdataset_tasks_registered

# Test full integration
XXX_RUN_INTEGRATION_TESTS=1 python -m unittest test.test_database_factory
```

## Test Organization

Tests use numbered prefixes to control execution order:

- `test_1_*` - Summary and overview tests
- `test_2_*` - Database registration tests
- `test_3_*` - Task registration tests (one per database)
- `test_4_*` - Error handling tests
- `test_5_*` - Registry structure tests

Within each test class, methods run alphabetically by name.

## Environment Variable

- `XXX_RUN_INTEGRATION_TESTS=0` (default): Run unit tests only, skip integration tests
- `XXX_RUN_INTEGRATION_TESTS=1`: Run integration tests only, skip unit tests

Replace `XXX` with your system name if needed.
