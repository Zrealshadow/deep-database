import collections
import random


class ModelScore:
    def __init__(self, model_id, score):
        self.model_id = model_id
        self.score = score

    def __repr__(self):
        return "m_{}_s_{}".format(self.model_id, self.score)


# for binary insert
def binary_insert_get_rank(rank_list: list, new_item: ModelScore) -> int:
    """
    Insert the new_item to rank_list, then get the rank of it.
    :param rank_list:
    :param new_item:
    :return:
    """
    index = search_position(rank_list, new_item)
    # search the position to insert into
    rank_list.insert(index, new_item)
    return index


# O(logN) search the position to insert into
def search_position(rank_list_m: list, new_item: ModelScore):
    if len(rank_list_m) == 0:
        return 0
    left = 0
    right = len(rank_list_m) - 1
    while left + 1 < right:
        mid = int((left + right) / 2)
        if rank_list_m[mid].score <= new_item.score:
            left = mid
        else:
            right = mid

    # consider the time.
    if rank_list_m[right].score <= new_item.score:
        return right + 1
    elif rank_list_m[left].score <= new_item.score:
        return left + 1
    else:
        return left


class Model(object):
    def __init__(self):
        self.arch = None
        self.score = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return "{:}".format(self.arch)


class RegularizedEASampler:

    def __init__(self, space, population_size: int, sample_size: int):

        self.population_size = population_size
        # list of object,
        self.population = collections.deque()
        # list of str, for duplicate checking
        self.population_model_ids = collections.deque()

        self.space = space
        self.sample_size = sample_size
        self.current_sampled = 0

        # id here is to match the outside value.
        self.current_arch_id = None
        self.current_arch_micro = None

        # use the visited to reduce the collapse
        self.visited = {}
        self.max_mutate_time = 4
        self.max_mutate_sampler_time = 4

    def sample_next_arch(self, sorted_model_ids: list):
        """
        This function performs one evolution cycle. It produces a model and removes another.
        Models are sampled randomly from the current population. If the population size is less than the
        desired population size, a random architecture is added to the population.

        :param sorted_model_ids: List of model ids sorted based on some criterion (not used here directly).
        :return: Tuple of the architecture id and the architecture configuration (micro).
        """
        # Case 1: If population hasn't reached desired size, add random architectures
        if len(self.population) < self.population_size:
            while True:
                arch_id, arch_micro = self.space.random_architecture_id()
                # Ensure that EA population has no repeated value
                if str(arch_id) not in self.population_model_ids:
                    break
            self.current_arch_micro = arch_micro
            self.current_arch_id = arch_id
            return arch_id, arch_micro

        # Case 2: If population has reached desired size, evolve population
        else:
            cur_mutate_sampler_time = 0
            is_found_new = False

            # Keep attempting mutations for a maximum of 'max_mutate_sampler_time' times
            while cur_mutate_sampler_time < self.max_mutate_sampler_time:
                cur_mutate_time = 0

                # Randomly select a sample of models from the population
                sample = []
                sample_ids = []
                while len(sample) < self.sample_size:
                    candidate = random.choice(list(self.population))
                    candidate_id = self.population_model_ids[self.population.index(candidate)]
                    sample.append(candidate)
                    sample_ids.append(candidate_id)

                # Select the best parent from the sample (based on the order in sorted_model_ids)
                parent_id = max(sample_ids, key=lambda _id: sorted_model_ids.index(str(_id)))
                parent = sample[sample_ids.index(parent_id)]

                # Try to mutate the parent up to 'max_mutate_time' times
                while cur_mutate_time < self.max_mutate_time:
                    arch_id, arch_micro = self.space.mutate_architecture(parent.arch)

                    # If the mutated architecture hasn't been visited or we've visited all possible architectures, stop
                    if arch_id not in self.visited or len(self.space) == len(self.visited):
                        self.visited[arch_id] = True
                        is_found_new = True
                        break
                    cur_mutate_time += 1

                # If we've found a new architecture, stop sampling
                if is_found_new:
                    break

                cur_mutate_sampler_time += 1

            # If we've hit the maximum number of mutation attempts, do nothing
            if cur_mutate_time * cur_mutate_sampler_time == self.max_mutate_time * self.max_mutate_sampler_time:
                pass

            # Update current architecture details
            self.current_arch_micro = arch_micro
            self.current_arch_id = arch_id

            return arch_id, arch_micro

    def fit_sampler(self, score: float):
        # if it's in Initialize stage, add to the population with random models.
        if len(self.population) < self.population_size:
            model = Model()
            model.arch = self.current_arch_micro
            model.score = score
            self.population.append(model)
            self.population_model_ids.append(self.current_arch_id)

        # if it's in mutation stage
        else:
            child = Model()
            child.arch = self.current_arch_micro
            child.score = score

            self.population.append(child)
            self.population_model_ids.append(self.current_arch_id)
            # Remove the oldest model.
            self.population.popleft()
            self.population_model_ids.popleft()


class SampleController:
    """
    Controller control the sample-score flow in the 1st phase.
    It records the results in the history.
    """

    def __init__(self, search_strategy):
        # Current ea is better than others.
        self.search_strategy = search_strategy

        # the large the index, the better the model
        self.ranked_models = []

        # when simple_score_sum=False, records the model's score of each algorithm,
        # use when simple_score_sum=True, record the model's sum score
        self.history = {}

    def sample_next_arch(self):
        """
        Return a generator
        :return:
        """
        return self.search_strategy.sample_next_arch(self.ranked_models)

    def fit_sampler(self, arch_id: str, alg_score: dict, simple_score_sum: bool = False,
                    is_sync: bool = True, arch_micro=None) -> float:
        """
        :param arch_id:
        :param alg_score: {alg_name1: score1, alg_name2: score2}
        :param simple_score_sum: if simply sum multiple scores (good performing),
                             or sum over their rank (worse performing)
        :return:
        """
        if simple_score_sum or len(alg_score.keys()) == 1:
            score = self._use_pure_score_as_final_res(arch_id, alg_score)
        else:
            score = self._use_vote_rank_as_final_res(arch_id, alg_score)
        if is_sync:
            self.search_strategy.fit_sampler(score)
        else:
            self.search_strategy.async_fit_sampler(arch_id, arch_micro, score)
        return score

    def _use_vote_rank_as_final_res(self, model_id: str, alg_score: dict):
        """
        :param model_id:
        :param alg_score: {alg_name1: score1, alg_name2: score2}
        """
        # todo: bug: only all scores' under all arg is greater than previous one, then treat it as greater.
        for alg in alg_score:
            if alg not in self.history:
                self.history[alg] = []

        # add model and score to local list
        for alg, score in alg_score.items():
            binary_insert_get_rank(self.history[alg], ModelScore(model_id, score))

        new_rank_score = self._re_rank_model_id(model_id, alg_score)
        return new_rank_score

    def _use_pure_score_as_final_res(self, model_id: str, alg_score: dict):
        # get the key and sum the score of various alg
        score_sum_key = "_".join(list(alg_score.keys()))
        if score_sum_key not in self.history:
            self.history[score_sum_key] = []
        final_score = 0
        for alg in alg_score:
            final_score += float(alg_score[alg])
        # insert and get rank
        index = binary_insert_get_rank(self.history[score_sum_key], ModelScore(model_id, final_score))
        self.ranked_models.insert(index, model_id)
        return final_score

    def _re_rank_model_id(self, model_id: str, alg_score: dict):
        # todo: re-rank everything, to make it self.ranked_models more accurate.
        model_new_rank_score = {}
        current_explored_models = 0
        for alg, score in alg_score.items():
            for rank_index in range(len(self.history[alg])):
                current_explored_models = len(self.history[alg])
                ms_ins = self.history[alg][rank_index]
                # rank = index + 1, since index can be 0
                if ms_ins.model_id in model_new_rank_score:
                    model_new_rank_score[ms_ins.model_id] += rank_index + 1
                else:
                    model_new_rank_score[ms_ins.model_id] = rank_index + 1

        for ele in model_new_rank_score.keys():
            model_new_rank_score[ele] = model_new_rank_score[ele] / current_explored_models

        self.ranked_models = [k for k, v in sorted(model_new_rank_score.items(), key=lambda item: item[1])]
        new_rank_score = model_new_rank_score[model_id]
        return new_rank_score

    def get_current_top_k_models(self, k=-1):
        """
        The model is already scored by: low -> high
        :param k:
        :return:
        """
        if k == -1:
            # retur all models
            return self.ranked_models
        else:
            return self.ranked_models[-k:]
