from torch.utils.data import DataLoader
from zero_cost_ms.search_space import init_search_space
from zero_cost_ms.search_alg.regularized_ea import RegularizedEASampler, SampleController
from zero_cost_ms.search_alg.evaluator import ModelAcquireData, FilterEvaluator
import json
import time


class ModelEvaData:
    """
    Eva worker send score to search strategy
    """

    def __init__(self, model_id: str = None, model_score: dict = None):
        if model_score is None:
            model_score = {}
        self.model_id = model_id
        self.model_score = model_score

    def serialize_model(self) -> str:
        data = {"model_id": self.model_id,
                "model_score": self.model_score}
        return json.dumps(data)

    @classmethod
    def deserialize(cls, data_str: str):
        data = json.loads(data_str)
        res = cls(
            data["model_id"],
            data["model_score"])
        return res


class FilteringEvaluator:

    def __init__(self, args, K: int, N: int, search_space_ins, train_loader: DataLoader = None):
        """
        Each model selection job will init one class here.
        :param args: space, population_size, sample_size
        :param K: K models return in 1st phase
        :param N: N models eval in total
        :param search_space_ins:
        """

        # return K models
        self.K = K
        # explore N models
        self.N = N
        self.args = args
        self.search_space_ins = search_space_ins

        if self.N >= min(len(self.search_space_ins), 100000):
            print("Explore all models")
            raise "too much"
        else:
            strategy = RegularizedEASampler(self.search_space_ins,
                                            population_size=self.args.population_size,
                                            sample_size=self.args.sample_size)
        self.sampler = SampleController(strategy)

        # seq: init the phase 1 evaluator,
        self._evaluator = FilterEvaluator(device=self.args.device,
                                          num_label=self.args.num_labels,
                                          dataset_name=self.args.dataset,
                                          search_space_ins=self.search_space_ins,
                                          train_loader=train_loader,
                                          metrics=self.args.tfmem)

    def run_phase1(self) -> (list, list, list, list):
        """
        Controller explore n models, and return the top K models.
        :return:
        """

        # those two are used to track performance trace
        # current best model id
        trace_highest_scored_models_id = []
        # current highest score
        trace_highest_score = []
        explored_n = 1
        model_eva = ModelEvaData()

        while explored_n <= self.N:
            # generate new model
            arch_id, arch_micro = self.sampler.sample_next_arch()
            # this is for sequence sampler.
            if arch_id is None:
                break
            model_encoding = self.search_space_ins.serialize_model_encoding(arch_micro)

            explored_n += 1

            # run the model selection
            model_acquire_data = ModelAcquireData(model_id=str(arch_id),
                                                  model_encoding=model_encoding,
                                                  is_last=False)
            data_str = model_acquire_data.serialize_model()

            model_eva.model_id = str(arch_id)
            model_eva.model_score = self._evaluator.filter_evaluate(data_str)

            if explored_n % 100 == 0:
                print("3. [trails] Phase 1: filter phase explored " + str(explored_n) +
                      " model, model_id = " + model_eva.model_id +
                      " model_scores = " + json.dumps(model_eva.model_score))

            ranked_score = self.sampler.fit_sampler(model_eva.model_id,
                                                    model_eva.model_score,
                                                    simple_score_sum=self.args.simple_score_sum)

            # this is to measure the value of metrix, sum of two value.
            if len(trace_highest_score) == 0:
                trace_highest_score.append(ranked_score)
                trace_highest_scored_models_id.append(str(arch_id))
            else:
                if ranked_score > trace_highest_score[-1]:
                    trace_highest_score.append(ranked_score)
                    trace_highest_scored_models_id.append(str(arch_id))
                else:
                    trace_highest_score.append(trace_highest_score[-1])
                    trace_highest_scored_models_id.append(trace_highest_scored_models_id[-1])

        print("3. [trails] Phase 1: filter phase explored " + str(explored_n) +
              " model, model_id = " + model_eva.model_id +
              " model_scores = " + json.dumps(model_eva.model_score))
        # return the top K models
        return self.sampler.get_current_top_k_models(self.K), self.sampler.get_current_top_k_models(-1), \
               trace_highest_score, trace_highest_scored_models_id


class RunModelSelection:

    def __init__(self, search_space_name: str, args):
        self.args = args

        # basic
        self.search_space_name = search_space_name
        self.dataset = self.args.dataset

        # instance of the search space.
        self.search_space_ins = init_search_space(search_space_name, self.args)

    def filtering_phase(self, N, K, train_loader=None):
        """
        Select model online for structured data.
        :param N:  explore N models.
        :param K:  keep K models
        :param train_loader:  data
        :return:
        """

        print("2. [trails] Begin filtering_phase...")
        begin_time = time.time()
        p1_runner = FilteringEvaluator(
            args=self.args,
            K=K, N=N,
            search_space_ins=self.search_space_ins,
            train_loader=train_loader)

        k_models, all_models, p1_trace_highest_score, p1_trace_highest_scored_models_id = p1_runner.run_phase1()
        print(f"2. [trails] filtering_phase Done, time_usage = {time.time() - begin_time}")
        return k_models, all_models, p1_trace_highest_score, p1_trace_highest_scored_models_id
