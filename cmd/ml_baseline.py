import torch_frame
import argparse

from torch_frame.gbdt import LightGBM, CatBoost
from torch_frame.typing import Metric
from sklearn.metrics import mean_absolute_error, roc_auc_score
from relbench.base import TaskType
from utils.data import TableData


parser = argparse.ArgumentParser(description="ML baseline for RelBench tasks.")


parser.add_argument("--data_dir", type=str, required=True,
                    help="Path to the data directory.")
parser.add_argument("--trials", type=int, default=10,
                    help="Number of trials for model training.")
parser.add_argument("--method", type=str, choices=["lgb", "catboost"],
                    default="lgb", help="Method to use for training.")
args = parser.parse_args()

data_dir = args.data_dir

# print the profile of this script
print(f"Test Table {data_dir} Using Model {args.method}")

table_data = TableData.load_from_dir(data_dir)

if table_data.task_type == TaskType.REGRESSION:
    tune_metrics = Metric.MAE
    task_type_ = torch_frame.TaskType.REGRESSION
    tm = mean_absolute_error
else:
    tune_metrics = Metric.ROCAUC
    task_type_ = torch_frame.TaskType.BINARY_CLASSIFICATION
    tm = roc_auc_score


if args.method == "lgb":
    model = LightGBM(
        task_type=task_type_,
        metric=tune_metrics,
    )
elif args.method == "catboost":
    model = CatBoost(
        task_type=task_type_,
        metric=tune_metrics,
    )

model.tune(tf_train=table_data.train_tf,
           tf_val=table_data.val_tf, num_trials=args.trials)


pred = model.predict(tf_test=table_data.test_tf).numpy()
y = table_data.test_tf.y.numpy()


eval_score = tm(y, pred)

print(f"Evaluation score ({args.method}): {tm.__name__}:{eval_score:.4f}")
