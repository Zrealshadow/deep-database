import json
import argparse
import os
import random
import numpy as np
import torch
from zero_cost_ms.data_loader import libsvm_dataloader
from zero_cost_ms.main import RunModelSelection
from utils.data import TableData
import torch_frame


def seed_everything(seed=2201):
    # 2022 -> 2021 -> 2031
    ''' [reference] https://gist.github.com/KirillVladimirov/005ec7f762293d2321385580d3dbe335 '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def sampler_args(parser):
    # define search space,
    parser.add_argument('--search_space', type=str, default="mlp_sp",
                        help='[nasbench101, nasbench201, mlp_sp]')
    # EA sampler's parameters,
    parser.add_argument('--population_size', type=int, default=10, help="The learning rate for REINFORCE.")
    parser.add_argument('--sample_size', type=int, default=3, help="The momentum value for EMA.")
    parser.add_argument('--simple_score_sum', default='True', type=str2bool,
                        help="Sum multiple TFMEM score or use Global Rank")


def mlp_args(parser):
    parser.add_argument('--num_layers', default=4, type=int, help='# hidden layers')
    parser.add_argument('--hidden_choice_len', default=20, type=int, help=
    'number of hidden layer choices, 10 for criteo, 20 for others')


def mlp_trainner_args(parser):
    parser.add_argument('--epoch', type=int, default=20,
                        help='number of maximum epochs, '
                             'frappe: 20, uci_diabetes: 40, criteo: 10'
                             'nb101: 108, nb201: 200')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help="learning reate")
    parser.add_argument('--patience', type=int, default=1, help='number of epochs for stopping training')
    # parser.add_argument('--eval_freq', type=int, default=10000, help='max number of batches to train per epoch')

    parser.add_argument('--iter_per_epoch', type=int, default=200,
                        help="None, "
                             "200 for frappe, uci_diabetes, 2000 for criteo")

    # MLP model config
    parser.add_argument('--nfeat', type=int, default=5500,
                        help='the number of features, '
                             'frappe: 5500, '
                             'uci_diabetes: 369,'
                             'criteo: 2100000')
    parser.add_argument('--nfield', type=int, default=10,
                        help='the number of fields, '
                             'frappe: 10, '
                             'uci_diabetes: 43,'
                             'criteo: 39')
    parser.add_argument('--nemb', type=int, default=10,
                        help='embedding size 10')

    # MLP train config
    parser.add_argument('--report_freq', type=int, default=30, help='report frequency')
    parser.add_argument('--workers', default=1, type=int, help='data loading workers')


def data_set_config(parser):
    parser.add_argument('--base_dir', type=str, default="../exp_data/",
                        help='path of data and result parent folder')
    # define search space,
    parser.add_argument('--dataset', type=str, default='frappe',
                        help='cifar10, cifar100, ImageNet16-120 '
                             'frappe, criteo, uci_diabetes')

    parser.add_argument('--num_labels', type=int, default=2,
                        help='[10, 100, 120],'
                             '[2, 2, 2]')


def system_performance_exp(parser):
    parser.add_argument('--models_explore', default=10, type=int, help='# models to explore in the filtering phase')
    parser.add_argument('--tfmem', default="ExpressFlow", type=str, help='the matrix t use, all_matrix')
    parser.add_argument('--embedding_cache_filtering', default='True', type=str2bool,
                        help='Cache embedding for MLP in filtering phase?')
    parser.add_argument('--concurrency', default=1, type=int, help='number of worker in filtering phase')


def parse_arguments():
    parser = argparse.ArgumentParser(description='system')

    # job config
    parser.add_argument('--log_name', type=str, default="main_T_100s")

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--device', type=str, default="cpu")

    parser.add_argument('--result_dir', default="./internal/ml/model_selection/exp_result/", type=str,
                        help='path to store exp outputs')
    parser.add_argument('--num_points', default=12, type=int, help='num GPus')

    parser.add_argument("--data_dir", type=str, required=False,
                        help="Path to the data directory.")

    sampler_args(parser)

    mlp_args(parser)
    data_set_config(parser)
    mlp_trainner_args(parser)
    system_performance_exp(parser)

    parser.add_argument('--max_load', type=int, default=-1, help="Max Loading time")

    seed_everything()

    return parser.parse_args()


def generate_data_loader(args):
    train_loader, val_loader, test_loader = libsvm_dataloader(
        args=args,
        data_dir="sample_data/frappe",
        nfield=args.nfield,
        batch_size=args.batch_size)
    class_num = args.num_labels
    return train_loader, val_loader, test_loader, class_num


def save_res(k_models, all_models):
    # Example: assuming k_models and all_models are dicts or lists
    data = {
        "k_models": k_models,
        "all_models": all_models
    }

    # Save to JSON file
    with open("result.json", "w") as f:
        json.dump(data, f, indent=4)  # indent makes it pretty-printed


if __name__ == "__main__":
    args = parse_arguments()
    args.data_dir = "/home/lingze/embedding_fusion/data/dfs-flatten-table/avito-ad-ctr"
    # train_loader, val_loader, test_loader, class_num = generate_data_loader(args)

    table_data = TableData.load_from_dir(args.data_dir)

    batch_size = 256
    data_loaders = {
        idx: torch_frame.data.DataLoader(
            getattr(table_data, f"{idx}_tf"),
            batch_size=batch_size,
            shuffle=idx == "train",
            pin_memory=True,
        )
        for idx in ["train", "val", "test"]
    }
    train_loader = data_loaders["train"]

    search_space = "mlp"
    rms = RunModelSelection(search_space, args)
    k_models, all_models, _, _ = rms.filtering_phase(1000, 100, train_loader)
    save_res(k_models, all_models)
