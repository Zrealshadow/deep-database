import torch
import math
import torch_frame
import argparse
import copy
import optuna
from tqdm import tqdm
import os

from torch_frame.nn.models import FTTransformer, ResNet
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
    
    # Common search space for all models
    common_space = {
        "channels": trial.suggest_categorical("channels", [64, 128, 256, 512]),
        "num_layers": trial.suggest_int("num_layers", 2, 6),
    }
    
    # Model-specific search spaces
    if model_name == "resnet":
        model_specific = {
            "dropout_prob": trial.suggest_float("dropout_prob", 0.1, 0.5, step=0.1),
            "normalization": trial.suggest_categorical("normalization", ["layer_norm", "batch_norm", "none"]),
        }
    elif model_name in ["fttransformer", "fttrans"]:
        # FTTransformer only uses channels and num_layers (architecture parameters)
        # out_channels is determined by the task, not searched
        model_specific = {}
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported models: ResNet, FTTransformer")
    
    return {**common_space, **model_specific}


def get_model_class(model_name):
    """Get model class based on model name"""
    model_name = model_name.lower()
    if model_name == "resnet":
        return ResNet
    elif model_name in ["fttransformer", "fttrans"]:
        return FTTransformer
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported models: ResNet, FTTransformer")


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
    
    # Add model-specific arguments
    model_name = model_name.lower()
    if model_name == "resnet":
        base_args.update({
            "dropout_prob": search_space["dropout_prob"],
            "normalization": search_space["normalization"],
        })
    elif model_name in ["fttransformer", "fttrans"]:
        # FTTransformer only uses channels and num_layers
        pass
    
    # Set out_channels based on task type (not searched)
    if is_regression:
        base_args["out_channels"] = 1  # Regression: single output
    else:
        # Classification: out_channels = number of classes
        # This should be determined from the dataset, not searched
        base_args["out_channels"] = 1  # For binary classification, or get from dataset
    
    return base_args


def objective(trial, table_data, is_regression, evaluate_matric_func, higher_is_better, model_name):
    """Optuna objective function for hyperparameter optimization"""

    # Get search space based on model type and task type
    search_space = get_search_space(trial, model_name, is_regression)

    # Fixed training parameters (not searched)
    batch_size = 256
    lr = 0.001
    num_epochs = 100  # Reduced for hyperparameter search
    early_stop_threshold = 5
    max_round_epoch = 10

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

        # Load best model and evaluate on test set
        if best_model_state is not None:
            net.load_state_dict(best_model_state)

        test_logits, _, test_pred_hat = test(
            net, data_loaders["test"], is_regression=is_regression)
        test_metric = evaluate_matric_func(test_pred_hat, test_logits)

        return test_metric

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
    parser.add_argument("--model", type=str, choices=["ResNet", "FTTransformer"], required=True,
                        help="Model architecture to optimize.")
    parser.add_argument("--n_trials", type=int, default=100,
                        help="Number of trials for hyperparameter optimization.")
    parser.add_argument("--study_name", type=str, default=None,
                        help="Name for the Optuna study (defaults to {model}_hyperopt).")

    args = parser.parse_args()

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

    # Best practice defaults: Use HyperbandPruner and SQLite storage
    pruner = optuna.pruners.HyperbandPruner()
    direction = "maximize" if higher_is_better else "minimize"

    # Create studies directory if it doesn't exist
    os.makedirs("studies", exist_ok=True)

    # Use SQLite storage for persistence (best practice)
    storage_path = f"sqlite:///studies/{args.study_name}.db"
    print(f"Using storage: {storage_path}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_path,
        load_if_exists=True,
        direction=direction,
        pruner=pruner
    )

    # Print search space based on model
    # Display search space based on model and task type
    if args.model.lower() == "resnet":
        print(f"Search space: channels=[64,128,256,512], "
              f"num_layers=[2,6], "
              f"dropout_prob=[0.1,0.5], "
              f"normalization=[layer_norm,batch_norm,none]")
    elif args.model.lower() in ["fttransformer", "fttrans"]:
        print(f"Search space: channels=[64,128,256,512], "
              f"num_layers=[2,6]")
    else:
        print(f"Unknown model: {args.model}")

    print(f"Starting hyperparameter optimization with {args.n_trials} trials...")

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, table_data, is_regression, evaluate_matric_func, higher_is_better, args.model),
        n_trials=args.n_trials
    )

    # Print results
    print("\n" + "=" * 50)
    print("HYPERPARAMETER OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

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

    print(f"\nStudy saved to: {storage_path}")
    print("You can resume this study by running the same command again!")


if __name__ == "__main__":
    main()

"""
USAGE EXAMPLES:

# ResNet hyperparameter optimization
python hyperopt_baseline.py --data_dir /home/lingze/embedding_fusion/data/dfs-flatten-table/avito-ad-ctr --model ResNet --n_trials 100 --study_name "resnet_study"

# FTTransformer hyperparameter optimization  
python hyperopt_baseline.py --data_dir /home/lingze/embedding_fusion/data/dfs-flatten-table/avito-ad-ctr --model FTTransformer --n_trials 100 --study_name "fttrans_study"


RESULTS:
- Results are automatically saved to: studies/{study_name}.db
- You can resume interrupted studies by running the same command again
- Best hyperparameters are printed at the end
"""
