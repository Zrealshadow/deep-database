"""
TODO: merge EMA class with update_moving_average() function.
"""
import torch
import numpy as np


class EMA:
    def __init__(self, beta:float, epochs:int):
        """
        beta: moving average decay rate
        epochs: total number of training epochs
        """
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new


def update_moving_average(ema_updater:EMA, ma_model:torch.nn.Module, current_model:torch.nn.Module):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)