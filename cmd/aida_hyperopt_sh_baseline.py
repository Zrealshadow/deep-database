import torch
import math
import torch_frame
import argparse
import copy
import optuna
from tqdm import tqdm
import os
import csv
import time
from datetime import datetime
from pathlib import Path

from torch_frame.nn.models import FTTransformer
from qzero.search_space.mlp import QZeroMLP
from qzero.search_space.resnet import QZeroResNet
from torch.nn import L1Loss, BCEWithLogitsLoss
from sklearn.metrics import mean_absolute_error, roc_auc_score
from relbench.base import TaskType

from model.base import construct_stype_encoder_dict, default_stype_encoder_cls_kwargs
from utils.resource import get_text_embedder_cfg
from utils.data import TableData

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")


def deactivate_dropout(net: torch.nn.Module):
    """Deactivate dropout layers in the model for regression task"""
    deactive_nn_instances = (
        torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)
    for module in net.modules():
        if isinstance(module, deactive_nn_instances):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def test(net: torch.nn.Module, loader: torch.utils.data.DataLoader, early_stop: int = -1, is_regression: bool = False):
    """Test function for model evaluation"""
    pred_list = []
    y_list = []
    early_stop = early_stop if early_stop > 0 else len(loader.dataset)

    if not is_regression:
        net.eval()

    for idx, batch in tqdm(enumerate(loader), total=len(loader), leave=False, desc="Testing"):
        with torch.no_grad():
            batch = batch.to(device)
            y = batch.y.float()
            pred = net(batch)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            pred_list.append(pred.detach().cpu())
            y_list.append(y.detach().cpu())
        if idx > early_stop:
            break
    pred_list = torch.cat(pred_list, dim=0)
    pred_logits = pred_list
    pred_list = torch.sigmoid(pred_list)
    y_list = torch.cat(y_list, dim=0).numpy()
    return pred_logits.numpy(), pred_list.numpy(), y_list


def get_search_space(trial, model_name, is_regression):
    """Get hyperparameter search space based on model type and task type"""
    model_name = model_name.lower()

    # Model-specific search spaces
    if model_name == "mlp":
        # QZeroMLP: get search space from model class
        channel_choices = QZeroMLP.channel_choices
        blocks_choices = QZeroMLP.blocks_choices
        num_layers = trial.suggest_categorical("num_layers", blocks_choices)

        channels = trial.suggest_categorical("channels", channel_choices)
        hidden_dims = [
            trial.suggest_categorical(f"hidden_dim_layer_{i}", channel_choices)
            for i in range(num_layers - 1)
        ]
        model_specific = {
            "channels": channels,
            "num_layers": num_layers,
            "hidden_dims": hidden_dims,
        }
    elif model_name == "resnet":
        # QZeroResNet: get search space from model class
        channel_choices = QZeroResNet.channel_choices
        blocks_choices = QZeroResNet.blocks_choices
        num_layers = trial.suggest_categorical("num_layers", blocks_choices)

        channels = trial.suggest_categorical("channels", channel_choices)
        block_widths = [
            trial.suggest_categorical(f"block_width_layer_{i}", channel_choices)
            for i in range(num_layers)
        ]
        model_specific = {
            "channels": channels,
            "num_layers": num_layers,
            "block_widths": block_widths,
        }
    elif model_name in ["fttransformer", "fttrans"]:
        # FTTransformer: uniform channels (original torch_frame model)
        num_layers = trial.suggest_int("num_layers", 2, 8)
        model_specific = {
            "channels": trial.suggest_categorical("channels", [32, 64, 128, 256]),
            "num_layers": num_layers,
        }
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported models: MLP, ResNet, FTTransformer")

    return model_specific


def get_model_class(model_name):
    """Get model class based on model name"""
    model_name = model_name.lower()
    if model_name == "mlp":
        return QZeroMLP
    elif model_name == "resnet":
        return QZeroResNet
    elif model_name in ["fttransformer", "fttrans"]:
        return FTTransformer
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported models: MLP, ResNet, FTTransformer")


def train_final_model(best_params_dict, model_name, table_data, is_regression, evaluate_matric_func, higher_is_better):
    """Train the final model with best hyperparameters (matching dnn_baseline_table_data.py config)"""

    print("\n" + "=" * 50)
    print("FINAL TRAINING WITH BEST HYPERPARAMETERS")
    print("=" * 50)
    print("Training configuration (from dnn_baseline_table_data.py):")
    print("  num_epochs: 200")
    print("  early_stop_threshold: 10")
    print("  batch_size: 256")
    print("  lr: 0.001")
    print("  max_round_epoch: 20")

    final_train_start = time.time()

    try:
        # Build final model with best hyperparameters
        final_search_space = get_search_space_from_params(best_params_dict, model_name)
        final_stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
        final_model_class = get_model_class(model_name)
        final_model_args = get_model_args(final_search_space, model_name, table_data, final_stype_encoder_dict,
                                          is_regression)
        final_net = final_model_class(**final_model_args)

        # Training configuration (matching dnn_baseline_table_data.py)
        final_batch_size = 256
        final_lr = 0.001
        final_num_epochs = 200
        final_early_stop_threshold = 10
        final_max_round_epoch = 20

        # Setup loss and optimizer
        if is_regression:
            final_loss_fn = L1Loss()
            deactivate_dropout(final_net)
        else:
            final_loss_fn = BCEWithLogitsLoss()

        final_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, final_net.parameters()), lr=final_lr)

        # Setup data loaders
        final_data_loaders = {
            idx: torch_frame.data.DataLoader(
                getattr(table_data, f"{idx}_tf"),
                batch_size=final_batch_size,
                shuffle=idx == "train",
                pin_memory=True,
            )
            for idx in ["train", "val", "test"]
        }

        final_net.to(device)
        final_patience = 0
        final_best_val_metric = -math.inf if higher_is_better else math.inf
        final_best_model_state = None

        # Training loop
        print("\nTraining...")
        for epoch in range(final_num_epochs):
            final_net.train()
            loss_accum = 0
            count_accum = 0

            for idx, batch in enumerate(final_data_loaders["train"]):
                if idx > final_max_round_epoch:
                    break

                final_optimizer.zero_grad()
                batch = batch.to(device)
                pred = final_net(batch)
                pred = pred.view(-1) if pred.size(1) == 1 else pred
                y = batch.y.float()
                loss = final_loss_fn(pred, y)

                loss.backward()
                final_optimizer.step()
                loss_accum += loss.item()
                count_accum += 1

            # Validation
            val_logits, _, val_pred_hat = test(
                final_net, final_data_loaders["val"], is_regression=is_regression)
            val_metric = evaluate_matric_func(val_pred_hat, val_logits)

            # Early stopping
            if (higher_is_better and val_metric > final_best_val_metric) or \
                    (not higher_is_better and val_metric < final_best_val_metric):
                final_best_val_metric = val_metric
                final_best_model_state = copy.deepcopy(final_net.state_dict())
                final_patience = 0
            else:
                final_patience += 1
                if final_patience > final_early_stop_threshold:
                    print(f"  Early stopped at epoch {epoch}")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}: val_metric={val_metric:.4f}")

        # Load best model and evaluate on test set
        if final_best_model_state:
            final_net.load_state_dict(final_best_model_state)

        test_logits, _, test_pred_hat = test(
            final_net, final_data_loaders["test"], is_regression=is_regression)
        test_metric = evaluate_matric_func(test_pred_hat, test_logits)

        final_train_end = time.time()
        final_train_time_seconds = final_train_end - final_train_start

        print(f"\n✅ Final training completed!")
        print(f"   Best validation metric: {final_best_val_metric:.6f}")
        print(f"   Test metric: {test_metric:.6f}")
        print(f"   Training time: {final_train_time_seconds:.2f} seconds ({final_train_time_seconds / 3600:.2f} hours)")

        return {
            "final_best_val_metric": final_best_val_metric,
            "final_test_metric": test_metric,
            "final_train_time_seconds": final_train_time_seconds,
        }

    except Exception as e:
        print(f"\n❌ Final training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def get_search_space_from_params(params_dict, model_name):
    """Reconstruct search_space dict from Optuna trial params"""
    model_name = model_name.lower()

    if model_name == "mlp" or model_name == "resnet":
        # Reconstruct the search space structure
        search_space = {
            "channels": params_dict["channels"],
            "num_layers": params_dict["num_layers"],
        }

        if model_name == "mlp":
            # Reconstruct hidden_dims list
            hidden_dims = []
            for i in range(params_dict["num_layers"] - 1):
                hidden_dims.append(params_dict[f"hidden_dim_layer_{i}"])
            search_space["hidden_dims"] = hidden_dims
        elif model_name == "resnet":
            # Reconstruct block_widths list
            block_widths = []
            for i in range(params_dict["num_layers"]):
                block_widths.append(params_dict[f"block_width_layer_{i}"])
            search_space["block_widths"] = block_widths

    elif model_name in ["fttransformer", "fttrans"]:
        search_space = {
            "channels": params_dict["channels"],
            "num_layers": params_dict["num_layers"],
        }

    return search_space


def get_model_args(search_space, model_name, table_data, stype_encoder_dict, is_regression):
    """Get model arguments based on search space and model type"""
    # Base arguments for all models
    base_args = {
        "channels": search_space["channels"],
        "num_layers": search_space["num_layers"],
        "col_names_dict": table_data.col_names_dict,
        "stype_encoder_dict": stype_encoder_dict,
        "col_stats": table_data.col_stats,
    }

    # Set out_channels based on task type (not searched)
    if is_regression:
        base_args["out_channels"] = 1  # Regression: single output
    else:
        base_args["out_channels"] = 1  # Binary classification

    # Add model-specific arguments
    model_name = model_name.lower()
    if model_name == "mlp":
        # MLP: num_layers + 1 to match ResNet's configurable layers
        # because hidden_dims = num_layers - 1, so we need num_layers + 1 to get num_layers hidden_dims
        base_args["num_layers"] = search_space["num_layers"] + 1
        base_args.update({
            "hidden_dims": search_space["hidden_dims"],
        })
    elif model_name == "resnet":
        base_args.update({
            "block_widths": search_space["block_widths"],
        })
    elif model_name in ["fttransformer", "fttrans"]:
        # FTTransformer: only uses channels and num_layers
        pass

    return base_args


def model_selection(trial, table_data, is_regression, evaluate_matric_func, higher_is_better, model_name):
    """Model selection objective function for hyperparameter optimization"""

    # Get search space based on model type and task type
    search_space = get_search_space(trial, model_name, is_regression)

    # Fixed training parameters (not searched)
    batch_size = 256
    lr = 0.001
    num_epochs = 100  # Reduced for hyperparameter search
    early_stop_threshold = 5
    max_round_epoch = 20

    try:
        # Construct model with suggested hyperparameters
        stype_encoder_dict = construct_stype_encoder_dict(
            default_stype_encoder_cls_kwargs,
        )

        # Get model class and arguments using modular functions
        model_class = get_model_class(model_name)
        model_args = get_model_args(search_space, model_name, table_data, stype_encoder_dict, is_regression)

        # Create model instance
        net = model_class(**model_args)

        # Setup loss and evaluation
        if is_regression:
            loss_fn = L1Loss()
            deactivate_dropout(net)
        else:
            loss_fn = BCEWithLogitsLoss()

        # Setup optimizer
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

        # Setup data loaders
        data_loaders = {
            idx: torch_frame.data.DataLoader(
                getattr(table_data, f"{idx}_tf"),
                batch_size=batch_size,
                shuffle=idx == "train",
                pin_memory=True,
            )
            for idx in ["train", "val", "test"]
        }

        net.to(device)
        patience = 0
        best_val_metric = -math.inf if higher_is_better else math.inf
        best_model_state = None

        # Training loop
        for epoch in range(num_epochs):
            loss_accum = count_accum = 0
            net.train()

            for idx, batch in tqdm(enumerate(data_loaders["train"]),
                                   leave=False,
                                   total=len(data_loaders["train"]),
                                   desc=f"Trial {trial.number} Epoch {epoch}"):

                if idx > max_round_epoch:
                    break

                optimizer.zero_grad()
                batch = batch.to(device)
                pred = net(batch)
                pred = pred.view(-1) if pred.size(1) == 1 else pred
                y = batch.y.float()
                loss = loss_fn(pred, y)

                loss.backward()
                optimizer.step()
                loss_accum += loss.item()
                count_accum += 1

            # Validation
            val_logits, _, val_pred_hat = test(
                net, data_loaders["val"], is_regression=is_regression)
            val_metric = evaluate_matric_func(val_pred_hat, val_logits)

            # Early stopping and model selection
            if (higher_is_better and val_metric > best_val_metric) or \
                    (not higher_is_better and val_metric < best_val_metric):
                best_val_metric = val_metric
                best_model_state = copy.deepcopy(net.state_dict())
                patience = 0
            else:
                patience += 1
                if patience > early_stop_threshold:
                    break

            # Report intermediate result for pruning
            trial.report(val_metric, epoch)

            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Return validation performance for hyperparameter optimization
        return best_val_metric

    except optuna.exceptions.TrialPruned:
        # This is normal pruning behavior, not an error
        raise  # Re-raise to let Optuna handle it properly
    except Exception as e:
        print(f"Trial failed with error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return float('inf') if not higher_is_better else float('-inf')


def main():
    parser = argparse.ArgumentParser(description="Model hyperparameter optimization using Optuna")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the data directory.")
    parser.add_argument("--model", type=str, choices=["MLP", "ResNet", "FTTransformer"], required=True,
                        help="Model architecture to optimize.")
    parser.add_argument("--n_trials", type=int, default=100,
                        help="Number of trials for hyperparameter optimization.")
    parser.add_argument("--study_name", type=str, default=None,
                        help="Name for the Optuna study (defaults to {model}_hyperopt).")
    parser.add_argument("--output_csv", type=str, default="results/hyperopt/hyperopt_results.csv",
                        help="Path to output CSV file for results (default: results/hyperopt/hyperopt_results.csv).")

    args = parser.parse_args()

    # Record start time
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Extract dataset name from data_dir
    dataset_name = Path(args.data_dir).name

    # Set default study name if not provided
    if args.study_name is None:
        args.study_name = f"{args.model.lower()}_hyperopt"

    # Load data
    print(f"Loading data from {args.data_dir}")
    table_data = TableData.load_from_dir(args.data_dir)

    if not table_data.is_materialize:
        text_cfg = get_text_embedder_cfg(device="cpu")
        table_data.materilize(col_to_text_embedder_cfg=text_cfg)

    # Setup task-specific parameters
    if table_data.task_type == TaskType.REGRESSION:
        evaluate_matric_func = mean_absolute_error
        higher_is_better = False
        is_regression = True
    else:
        evaluate_matric_func = roc_auc_score
        higher_is_better = True
        is_regression = False

    print(f"Model: {args.model}")
    print(f"Task type: {table_data.task_type}")
    print(f"Optimization direction: {'maximize' if higher_is_better else 'minimize'}")

    # Use HyperbandPruner and in-memory storage (no database file)
    pruner = optuna.pruners.HyperbandPruner()
    direction = "maximize" if higher_is_better else "minimize"

    # Use in-memory storage (no files created)
    print(f"Using in-memory storage (no database file will be created)")

    study = optuna.create_study(
        study_name=args.study_name,
        direction=direction,
        pruner=pruner
    )

    # Print search space based on model
    # Display search space based on model and task type
    if args.model.lower() == "mlp":
        print(f"Search space (q-zero): channels={QZeroMLP.channel_choices}, "
              f"num_layers={QZeroMLP.blocks_choices}, "
              f"hidden_dims=per-layer from {QZeroMLP.channel_choices}")
    elif args.model.lower() == "resnet":
        print(f"Search space (q-zero): channels={QZeroResNet.channel_choices}, "
              f"num_layers={QZeroResNet.blocks_choices}, "
              f"block_widths=per-layer from {QZeroResNet.channel_choices}")
    elif args.model.lower() in ["fttransformer", "fttrans"]:
        print(f"Search space (original): channels=[64,128,256,512], "
              f"num_layers=[2,8]")
    else:
        print(f"Unknown model: {args.model}")

    print(f"Starting hyperparameter optimization with {args.n_trials} trials...")

    # Run optimization
    study.optimize(
        lambda trial: model_selection(trial, table_data, is_regression, evaluate_matric_func, higher_is_better,
                                      args.model),
        n_trials=args.n_trials
    )

    # Calculate model selection time
    selection_end_time = time.time()
    selection_time_seconds = selection_end_time - start_time

    # Print model selection results
    print("\n" + "=" * 50)
    print("HYPERPARAMETER OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Model selection time: {selection_time_seconds:.2f} seconds ({selection_time_seconds / 3600:.2f} hours)")

    # Prepare result data
    result_data = {
        "timestamp": timestamp,
        "dataset": dataset_name,
        "architecture": args.model,
        "n_trials": args.n_trials,
        "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "selection_time_seconds": f"{selection_time_seconds:.2f}",
        "best_val_metric": None,
        "best_params": None,
        "final_best_val_metric": None,
        "final_test_metric": None,
        "final_train_time_seconds": None,
        "total_time_seconds": None,
        "metric": "roc_auc" if higher_is_better else "mae",
    }

    best_params_dict = None
    if study.trials:
        best_trial = study.best_trial
        print(f"\nBest trial:")
        print(f"  Value: {best_trial.value:.6f}")
        print(f"  Params:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        print(f"\nBest hyperparameters for {args.model}:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")

        result_data["best_val_metric"] = f"{best_trial.value:.6f}"
        result_data["best_params"] = str(best_trial.params)
        best_params_dict = best_trial.params

    # ========== Final Training with Best Hyperparameters ==========
    if best_params_dict:
        final_results = train_final_model(
            best_params_dict,
            args.model,
            table_data,
            is_regression,
            evaluate_matric_func,
            higher_is_better
        )

        if final_results:
            result_data["final_best_val_metric"] = f"{final_results['final_best_val_metric']:.6f}"
            result_data["final_test_metric"] = f"{final_results['final_test_metric']:.6f}"
            result_data["final_train_time_seconds"] = f"{final_results['final_train_time_seconds']:.2f}"

    # Calculate total time
    total_end_time = time.time()
    total_time_seconds = total_end_time - start_time
    result_data["total_time_seconds"] = f"{total_time_seconds:.2f}"

    print(f"\n{'=' * 50}")
    print(f"TOTAL TIME: {total_time_seconds:.2f} seconds ({total_time_seconds / 3600:.2f} hours)")
    print(f"{'=' * 50}")

    # Save results to CSV
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Check if CSV exists to determine if we need to write header
    file_exists = output_csv.exists()

    with open(output_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_data)

    print(f"\n✅ Results appended to: {output_csv}")
    print(f"   Dataset: {dataset_name}, Model: {args.model}, Time: {total_time_seconds:.2f}s")


if __name__ == "__main__":
    main()

"""
USAGE EXAMPLES:

# MLP hyperparameter optimization (q-zero with per-layer channels)
python aida_hyperopt_sh_baseline.py --data_dir /home/lingze/embedding_fusion/data/dfs-flatten-table/avito-ad-ctr --model MLP --n_trials 50 --study_name "mlp_avito"

# ResNet hyperparameter optimization (q-zero with per-block widths)
python aida_hyperopt_sh_baseline.py --data_dir /home/lingze/embedding_fusion/data/dfs-flatten-table/event-user-attendance --model ResNet --n_trials 50 --study_name "resnet_event"

# FTTransformer hyperparameter optimization (original torch_frame model)
python aida_hyperopt_sh_baseline.py --data_dir /home/lingze/embedding_fusion/data/dfs-flatten-table/ratebeer-user-active --model FTTransformer --n_trials 50 --study_name "fttrans_ratebeer"

# Custom output CSV file
python aida_hyperopt_sh_baseline.py --data_dir /path/to/data --model MLP --n_trials 50 --output_csv results/my_experiments.csv


SEARCH SPACE:
- MLP (q-zero): blocks=[2,3], channels=[64,128,256], per-layer hidden_dims (108 configs)
- ResNet (q-zero): blocks=[2,3], channels=[64,128,256], per-layer block_widths (108 configs)
- FTTransformer (original): channels=[64,128,256,512], num_layers=[2,8], uniform channels (28 configs)
- NO dropout or normalization search (using model defaults: dropout=0.2, norm=layer_norm)

OUTPUTS:
1. CSV Results File: results/hyperopt/hyperopt_results.csv (default)
   - Automatically appends one row per experiment
   - Columns: timestamp, dataset, architecture, n_trials, n_completed, n_pruned,
             selection_time_seconds, best_val_metric, best_params,
             final_best_val_metric, final_test_metric, final_train_time_seconds,
             total_time_seconds, metric
   - Perfect for tracking multiple experiments across datasets

WORKFLOW (两个独立阶段，无交叉):
1. Phase 1: Model Selection (超参数搜索)
   - Time: selection_time_seconds
   - Config: 100 epochs, early_stop=5, max_round_epoch=20
   - Output: best_val_metric (搜索阶段验证集最佳)
   - Output: best_params (最佳超参数)
   
2. Phase 2: Final Training (用最佳超参数重新训练)
   - Time: final_train_time_seconds
   - Config: 200 epochs, early_stop=10, max_round_epoch=20 (matching dnn_baseline_table_data.py)
   - Output: final_best_val_metric (完整训练验证集最佳，Early Stop的基础)
   - Output: final_test_metric (测试集性能)
   
3. Total: total_time_seconds = selection_time_seconds + final_train_time_seconds

EXAMPLE CSV OUTPUT:
timestamp,dataset,architecture,n_trials,n_completed,n_pruned,selection_time_seconds,best_val_metric,best_params,final_best_val_metric,final_test_metric,final_train_time_seconds,total_time_seconds,metric
2025-01-22 14:30:52,avito-ad-ctr,ResNet,50,32,18,12345.67,0.8567,"{'channels': 128}",0.8623,0.8645,3456.78,15802.45,roc_auc
"""
