from torch.utils.data import DataLoader
import time
import torch
from torch import nn
from .proxies import evaluator_register


class ModelAcquireData:
    """
    Eva worker get model from search strategy
    The serialize/deserialize is for good scalability. The project can be decouple into multiple service
    """

    def __init__(self, model_id: str, model_encoding: str, is_last: bool = False,
                 spi_seconds=None, spi_mini_batch=None, batch_size=32):
        self.is_last = is_last
        self.model_id = model_id
        self.model_encoding = model_encoding

        # this is when using spi
        self.spi_seconds = spi_seconds
        self.spi_mini_batch = spi_mini_batch
        self.batch_size = batch_size

    def serialize_model(self) -> dict:
        data = {"is_last": self.is_last,
                "model_id": self.model_id,
                "model_encoding": self.model_encoding,
                "spi_seconds": self.spi_seconds,
                "preprocess_seconds": self.spi_seconds,
                "batch_size": self.batch_size,
                "spi_mini_batch": self.spi_mini_batch}

        return data

    @classmethod
    def deserialize(cls, data: dict):
        res = cls(
            model_id=data["model_id"],
            model_encoding=data["model_encoding"],
            is_last=data["is_last"],
            spi_mini_batch=data["spi_mini_batch"],
            batch_size=data["batch_size"],
            spi_seconds=data["spi_seconds"])
        return res


class FilterEvaluator:

    def __init__(self, device: str, num_label: int, dataset_name: str,
                 search_space_ins,
                 train_loader: DataLoader, metrics: str = "ExpressFlow",
                 enable_cache: bool = False, data_retrievel: str = "sql"):
        """
        :param device:
        :param num_label:
        :param dataset_name:
        :param search_space_ins:
        :param search_space_ins:
        :param train_loader:
        :param metrics: which TFMEM to use?
        :param enable_cache: if cache embedding for scoring? only used on structued data
        :param data_retrievel: sql or spi
        """
        self.metrics = metrics
        # used only is_simulate = True
        self.score_getter = None

        # dataset settings
        self.dataset_name = dataset_name
        self.train_loader = train_loader
        self.num_labels = num_label

        self.search_space_ins = search_space_ins

        self.device = device

        # this is to do the expeirment
        self.enable_cache = enable_cache
        self.model_cache = None

        # performance records
        self.time_usage = {
            "model_id": [],

            "latency": 0.0,
            "io_latency": 0.0,
            "compute_latency": 0.0,

            "track_compute": [],  # compute time
            "track_io_model_init": [],  # init model weight
            "track_io_model_load": [],  # load model into GPU/CPU
            "track_io_res_load": [],  # load result into GPU/CPU
            "track_io_data_retrievel": [],  # release data
            "track_io_data_preprocess": [],  # pre-processing
        }

        self.last_id = -1
        self.data_retrievel = data_retrievel

        # at the benchmarking, we only use one batch for fast evaluate
        self.cached_mini_batch = None
        self.cached_mini_batch_target = None

        self.conn = None

    def if_cuda_avaiable(self):
        if "cuda" in self.device:
            return True
        else:
            return False

    def filter_evaluate(self, data_str: dict) -> dict:
        """
        :param data_str: encoded ModelAcquireData
        :return:
        """
        model_acquire = ModelAcquireData.deserialize(data_str)
        return self._filter_evaluate_online(model_acquire)

    def _filter_evaluate_online(self, model_acquire: ModelAcquireData) -> dict:

        model_encoding = model_acquire.model_encoding

        # 1. Get a batch of data
        mini_batch, mini_batch_targets, data_load_time_usage, data_pre_process_time = self.retrievel_data(model_acquire)

        self.time_usage["track_io_data_retrievel"].append(data_load_time_usage)

        # 2. Score
        if self.metrics == "PRUNE_SYNFLOW" or self.metrics == "ExpressFlow":
            bn = False
        else:
            bn = True
        # measure model load time
        begin = time.time()
        new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=bn)

        # mlp have embedding layer, which can be cached, optimization!
        if self.search_space_ins.name == "mlp":
            if self.enable_cache:
                new_model.init_embedding(self.model_cache)
                if self.model_cache is None:
                    self.model_cache = new_model.embedding.to(self.device)
            else:
                # init embedding every time created a new model
                new_model.init_embedding()

        self.time_usage["track_io_model_init"].append(time.time() - begin)

        if self.if_cuda_avaiable():
            begin = time.time()
            new_model = new_model.to(self.device)
            torch.cuda.synchronize()
            self.time_usage["track_io_model_load"].append(time.time() - begin)
        else:
            self.time_usage["track_io_model_load"].append(0)

        # measure data load time
        begin = time.time()
        mini_batch = self.data_pre_processing(mini_batch, self.metrics, new_model)
        self.time_usage["track_io_data_preprocess"].append(data_pre_process_time + time.time() - begin)

        _score, compute_time = evaluator_register[self.metrics].evaluate_wrapper(
            arch=new_model,
            device=self.device,
            space_name=self.search_space_ins.name,
            batch_data=mini_batch,
            batch_labels=mini_batch_targets)

        self.time_usage["track_compute"].append(compute_time)

        if self.if_cuda_avaiable():
            begin = time.time()
            _score = _score.item()
            torch.cuda.synchronize()
            self.time_usage["track_io_res_load"].append(time.time() - begin)

        else:
            _score = _score.item()
            self.time_usage["track_io_res_load"].append(0)

        model_score = {self.metrics: abs(_score)}
        del new_model
        return model_score

    def retrievel_data(self, model_acquire):
        if self.cached_mini_batch is None and self.cached_mini_batch_target is None:
            # this is structure data
            begin = time.time()
            batch = iter(self.train_loader).__next__()
            target = batch['y'].type(torch.LongTensor).to(self.device)
            batch['id'] = batch['id'].to(self.device)
            batch['value'] = batch['value'].to(self.device)

            # wait for moving data to GPU
            if self.if_cuda_avaiable():
                torch.cuda.synchronize()
            time_usage = time.time() - begin
            self.cached_mini_batch = batch
            self.cached_mini_batch_target = target
            return batch, target, time_usage, 0
        else:
            return self.cached_mini_batch, self.cached_mini_batch_target, 0, 0

    def data_pre_processing(self, mini_batch, metrics: str, new_model: nn.Module):

        # for those two metrics, we use all one embedding for efficiency (as in their paper)
        if metrics in ["ExpressFlow"]:
            if isinstance(mini_batch, torch.Tensor):
                feature_dim = list(mini_batch[0, :].shape)
                # add one dimension to feature dim, [1] + [3, 32, 32] = [1, 3, 32, 32]
                mini_batch = torch.ones([1] + feature_dim).float().to(self.device)
            else:
                # this is for the tabular data,
                mini_batch = new_model.generate_all_ones_embedding(4).float().to(self.device)
                # print(mini_batch.size())

        # wait for moving data to GPU
        if self.if_cuda_avaiable():
            torch.cuda.synchronize()
        return mini_batch
