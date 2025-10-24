import torch
import math
import torch_frame
import argparse
import copy
from tqdm import tqdm

from torch_frame.nn.models import MLP, ResNet, FTTransformer
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


def get_model_args(args, table_data, stype_encoder_dict, is_regression):
    """Get model arguments based on command line arguments"""
    # Base arguments for all models
    base_args = {
        "channels": args.channels,
        "num_layers": args.num_layers,
        "col_names_dict": table_data.col_names_dict,
        "stype_encoder_dict": stype_encoder_dict,
        "col_stats": table_data.col_stats,
    }

    # Add model-specific arguments
    model_name = args.model.lower()
    if model_name == "resnet":
        base_args.update({
            "dropout_prob": args.dropout_prob,
            "normalization": args.normalization,
        })
    elif model_name in ["fttransformer", "fttrans"]:
        # FTTransformer only uses channels and num_layers
        pass
    elif model_name == "mlp":
        base_args.update({
            "dropout_prob": args.dropout_prob,
            "normalization": args.normalization,
        })

    # Set out_channels based on task type
    if is_regression:
        base_args["out_channels"] = 1  # Regression: single output
    else:
        # Classification: out_channels = number of classes
        base_args["out_channels"] = 1  # For binary classification

    return base_args


def main():
    parser = argparse.ArgumentParser(description="Configurable DNN Baseline Training Script")

    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the data directory.")

    # Architecture parameters (configurable like hyperopt search space)
    parser.add_argument("--model", type=str, choices=["MLP", "ResNet", "FTTransformer"], required=True,
                        help="Model architecture type.")
    parser.add_argument("--channels", type=int, choices=[64, 128, 256, 512], default=128,
                        help="Number of channels (64, 128, 256, 512).")
    parser.add_argument("--num_layers", type=int, choices=range(2, 7), default=2,
                        help="Number of layers (2-6).")

    # Model-specific parameters
    parser.add_argument("--dropout_prob", type=float, choices=[0.1, 0.2, 0.3, 0.4, 0.5], default=0.2,
                        help="Dropout probability (0.1-0.5, step=0.1). Only used for MLP and ResNet.")
    parser.add_argument("--normalization", type=str, choices=["layer_norm", "batch_norm", "none"], default="layer_norm",
                        help="Normalization type. Only used for MLP and ResNet.")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="Number of training epochs.")
    parser.add_argument("--early_stop_threshold", type=int, default=10,
                        help="Number of epochs to wait for improvement before early stopping.")
    parser.add_argument("--max_round_epoch", type=int, default=20,
                        help="Maximum number of epochs per round.")

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_dir}")
    table_data = TableData.load_from_dir(args.data_dir)

    if not table_data.is_materialize:
        text_cfg = get_text_embedder_cfg(device="cpu")
        table_data.materilize(col_to_text_embedder_cfg=text_cfg)

    # Setup task-specific parameters
    if table_data.task_type == TaskType.REGRESSION:
        loss_fn = L1Loss()
        evaluate_matric_func = mean_absolute_error
        higher_is_better = False
        is_regression = True
    else:
        loss_fn = BCEWithLogitsLoss()
        evaluate_matric_func = roc_auc_score
        higher_is_better = True
        is_regression = False

    print(f"Model: {args.model}")
    print(f"Task type: {table_data.task_type}")
    print(f"Architecture parameters:")
    print(f"  channels: {args.channels}")
    print(f"  num_layers: {args.num_layers}")
    if args.model.lower() in ["mlp", "resnet"]:
        print(f"  dropout_prob: {args.dropout_prob}")
        print(f"  normalization: {args.normalization}")

    # Construct model
    stype_encoder_dict = construct_stype_encoder_dict(
        default_stype_encoder_cls_kwargs,
    )

    model_args = get_model_args(args, table_data, stype_encoder_dict, is_regression)

    # Create model instance
    model_name = args.model.lower()
    if model_name == "mlp":
        net = MLP(**model_args)
    elif model_name == "resnet":
        net = ResNet(**model_args)
    elif model_name in ["fttransformer", "fttrans"]:
        net = FTTransformer(**model_args)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Setup optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)

    # Setup data loaders
    data_loaders = {
        idx: torch_frame.data.DataLoader(
            getattr(table_data, f"{idx}_tf"),
            batch_size=args.batch_size,
            shuffle=idx == "train",
            pin_memory=True,
        )
        for idx in ["train", "val", "test"]
    }

    # Deactivate dropout for regression task
    if is_regression:
        deactivate_dropout(net)

    # Training loop
    net.to(device)
    patience = 0
    best_epoch = 0
    best_val_metric = -math.inf if higher_is_better else math.inf
    best_model_state = None

    print(f"\nStarting training with {args.num_epochs} epochs...")
    print(f"Early stopping threshold: {args.early_stop_threshold}")
    print(f"Max epochs per round: {args.max_round_epoch}")
    print("=" * 60)

    for epoch in range(args.num_epochs):
        loss_accum = count_accum = 0
        net.train()

        for idx, batch in tqdm(enumerate(data_loaders["train"]),
                               leave=False,
                               total=len(data_loaders["train"]),
                               desc=f"Epoch {epoch}"):

            if idx > args.max_round_epoch:
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

        train_loss = loss_accum / count_accum
        val_logits, _, val_pred_hat = test(
            net, data_loaders["val"], is_regression=is_regression)

        val_metric = evaluate_matric_func(val_pred_hat, val_logits)

        print(
            f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val {evaluate_matric_func.__name__}: {val_metric:.6f}")

        # Early stopping and model selection
        if (higher_is_better and val_metric > best_val_metric) or \
                (not higher_is_better and val_metric < best_val_metric):
            best_val_metric = val_metric
            best_epoch = epoch
            best_model_state = copy.deepcopy(net.state_dict())
            patience = 0

            test_logits, _, test_pred_hat = test(
                net, data_loaders["test"], is_regression=is_regression)
            test_metric = evaluate_matric_func(test_pred_hat, test_logits)

            print(f"  -> New best! Test {evaluate_matric_func.__name__}: {test_metric:.6f}")
        else:
            patience += 1
            if patience > args.early_stop_threshold:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final evaluation
    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    test_logits, _, test_pred_hat = test(
        net, data_loaders["test"], is_regression=is_regression)
    test_metric = evaluate_matric_func(test_pred_hat, test_logits)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation {evaluate_matric_func.__name__}: {best_val_metric:.6f}")
    print(f"Final test {evaluate_matric_func.__name__}: {test_metric:.6f}")
    
    # Save the best model
    from pathlib import Path
    
    # Create models directory if it doesn't exist
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Generate model filename based on dataset and model config
    dataset_name = Path(args.data_dir).name
    model_filename = f"{args.model}_{dataset_name}_ch{args.channels}_l{args.num_layers}"
    if args.model.lower() in ["mlp", "resnet"]:
        model_filename += f"_drop{args.dropout_prob}_norm{args.normalization}"
    model_filename += f"_val{best_val_metric:.6f}.pth"
    
    model_path = model_dir / model_filename
    
    # Save model state and metadata
    model_data = {
        'model_state_dict': best_model_state,
        'model_args': get_model_args(args, table_data, stype_encoder_dict, is_regression),
        'best_epoch': best_epoch,
        'best_val_metric': best_val_metric,
        'final_test_metric': test_metric,
        'dataset_name': dataset_name,
        'task_type': 'regression' if is_regression else 'classification',
        'evaluation_metric': evaluate_matric_func.__name__
    }
    
    torch.save(model_data, model_path)
    print(f"\nModel saved to: {model_path}")

    print("\nArchitecture used:")
    print(f"  Model: {args.model}")
    print(f"  Channels: {args.channels}")
    print(f"  Layers: {args.num_layers}")
    if args.model.lower() in ["mlp", "resnet"]:
        print(f"  Dropout: {args.dropout_prob}")
        print(f"  Normalization: {args.normalization}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
# ResNet with specific architecture
python ./cmd/dnn_configurable_baseline.py \
    --data_dir "/path/to/data" \
    --model ResNet \
    --channels 256 \
    --num_layers 4 \
    --dropout_prob 0.3 \
    --normalization batch_norm

# FTTransformer (only channels and layers matter)
python ./cmd/dnn_configurable_baseline.py \
    --data_dir "/path/to/data" \
    --model FTTransformer \
    --channels 128 \
    --num_layers 3
"""
