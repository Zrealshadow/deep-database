from functools import partial
import math
import time
from abc import abstractmethod
import torch
from torch import nn


class IntegratedHook:
    def __init__(self):
        self.originals = []
        self.perturbations = []
        self.Vs = []
        self.activation_map = {}
        self.is_perturbed = False

    def forward_hook(self, module, input, output):
        # Store the output based on whether it's perturbed or not
        if isinstance(module, nn.ReLU):
            if self.is_perturbed:
                self.perturbations.append(output)
            else:
                self.originals.append(output)

        # Save this output in the map using the module's ID
        self.activation_map[id(module)] = output

        # Register backward hook for gradient computation
        # todo: this will messed up the reference, result in the memory leak.
        # output.register_hook(lambda grad: self.backward_hook(grad, module))
        output.register_hook(partial(self.backward_hook, module=module))

    def backward_hook(self, grad, module):
        dz = grad  # gradient
        # Get the correct activation from the map
        activation = self.activation_map[id(module)]
        V = activation * abs(dz)  # product
        self.Vs.append(V)

    def calculate_trajectory_length(self, epsilon):
        # assert len(self.originals) == len(self.perturbations)
        trajectory_lengths = [abs(x_perturbed - x).norm() / epsilon for x, x_perturbed in
                              zip(self.originals, self.perturbations)]
        return trajectory_lengths

    def clear_all(self):
        self.originals.clear()
        self.perturbations.clear()
        self.Vs.clear()
        self.activation_map.clear()


class Evaluator:
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, arch: nn.Module,
                 device: str,
                 batch_data: object, batch_labels: torch.Tensor,
                 space_name: str
                 ) -> float:
        """
        Score each architecture with predefined architecture and data
        :param arch: architecture to be scored
        :param device:  cpu or gpu
        :param batch_data: a mini batch of data, [ batch_size, channel, W, H ] or dict for structure data
        :param batch_labels: a mini batch of labels
        :param space_name: string
        :return: score
        """
        raise NotImplementedError

    def evaluate_wrapper(self, arch, device: str, space_name: str,
                         batch_data: torch.tensor,
                         batch_labels: torch.tensor) -> (float, float):
        """
        :param arch: architecture to be scored
        :param device: cpu or GPU
        :param space_name: search space name
        :param batch_data: a mini batch of data, [ batch_size, channel, W, H ]
        :param batch_labels: a mini batch of labels
        :return: score, timeUsage
        """

        arch.train()
        arch.zero_grad()

        # measure scoring time
        if "cuda" in device:
            torch.cuda.synchronize()
            # use this will not need cuda.sync
            # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # starter.record()
            starter, ender = time.time(), time.time()
        else:
            starter, ender = time.time(), time.time()

        # score
        score = self.evaluate(arch, device, batch_data, batch_labels, space_name)

        if "cuda" in device:
            # ender.record()
            # implicitly waits for the event to be marked as complete before calculating the time difference
            # curr_time = starter.elapsed_time(ender)
            torch.cuda.synchronize()
            ender = time.time()
            curr_time = ender - starter
        else:
            ender = time.time()
            curr_time = ender - starter

        if math.isnan(score):
            if score > 0:
                score = 1e8
            else:
                score = -1e8
        if math.isinf(score):
            if score > 0:
                score = 1e8
            else:
                score = -1e8

        return score, curr_time


class ExpressFlowEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def linearize(self, arch):
        signs = {}
        for name, param in arch.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(self, arch, signs):
        for name, param in arch.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    def evaluate(self, arch: nn.Module, device, batch_data: object, batch_labels: torch.Tensor,
                 space_name: str) -> float:

        self.n_in = batch_data.shape[1]
        # Step 1: Linearize
        signs = self.linearize(arch.mlp)
        arch.mlp.double()

        hook_obj = IntegratedHook()
        hooks = []
        for module in arch.modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_forward_hook(hook_obj.forward_hook))

        epsilon = 1e-5
        delta_x = torch.randn_like(batch_data) * epsilon

        # Forward pass with original input
        hook_obj.is_perturbed = False
        out = arch.forward_wo_embedding(batch_data.double())

        # Forward pass with perturbed input
        hook_obj.is_perturbed = True
        _ = arch.forward_wo_embedding(batch_data.double() + delta_x)

        trajectory_lengths = hook_obj.calculate_trajectory_length(epsilon)

        # directly sum
        torch.sum(out).backward()

        total_sum = self.weighted_score(trajectory_lengths, hook_obj.Vs)

        # Step 2: Nonlinearize
        self.nonlinearize(arch.mlp, signs)

        # Remove the hooks
        for hook in hooks:
            hook.remove()
        del hooks
        hook_obj.clear_all()

        return total_sum

    def weighted_score(self, trajectory_lengths, Vs):
        trajectory_lengths.reverse()
        # Modify trajectory_lengths to ensure that deeper layers have smaller weights
        # For example, by taking the inverse of each computed trajectory length.
        inverse_trajectory_lengths = [1.0 / (length + 1e-6) for length in trajectory_lengths]

        # Normalize trajectory lengths if needed (this ensures the weights aren't too large)
        normalized_lengths = [length / sum(inverse_trajectory_lengths) for length in inverse_trajectory_lengths]

        # Use the normalized trajectory lengths as weights for your total_sum
        total_sum = sum(
            normalized_length * V.flatten().sum() * V.shape[1]
            for normalized_length, V in zip(normalized_lengths, Vs))
        total_sum = total_sum

        return total_sum

    def compute_sigma_w(self, n_in, n_out):
        factor = (6 / (n_in + n_out)) ** 0.5
        variance = (1 / 3) * factor ** 2
        sigma_w_squared = n_out * variance
        sigma_w = sigma_w_squared ** 0.5
        return sigma_w

    def _lower_bounded(self, n_in: int, k: int, d: int):
        return (self.compute_sigma_w(n_in, k) * math.sqrt(k) / math.sqrt(k + 1)) ** d


evaluator_register = {
    "ExpressFlow": ExpressFlowEvaluator(),

    # # sum on gradient
    # CommonVars.GRAD_NORM: GradNormEvaluator(),
    # CommonVars.GRAD_PLAIN: GradPlainEvaluator(),
    # #
    # # # training free matrix
    # # CommonVars.JACOB_CONV: JacobConvEvaluator(),
    # CommonVars.NAS_WOT: NWTEvaluator(),
    #
    # # this is ntk based
    # CommonVars.NTK_CONDNUM: NTKCondNumEvaluator(),
    # CommonVars.NTK_TRACE: NTKTraceEvaluator(),
    #
    # CommonVars.NTK_TRACE_APPROX: NTKTraceApproxEvaluator(),
    #
    # # # prune based
    # CommonVars.PRUNE_FISHER: FisherEvaluator(),
    # CommonVars.PRUNE_GRASP: GraspEvaluator(),
    # CommonVars.PRUNE_SNIP: SnipEvaluator(),
    # CommonVars.PRUNE_SYNFLOW: SynFlowEvaluator(),

}
