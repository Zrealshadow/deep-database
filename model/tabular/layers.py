import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, ninput, nlayers, nhid, dropout, noutput=1):
        super().__init__()
        layers = list()
        for i in range(nlayers):
            layers.append(nn.Linear(ninput, nhid))
            layers.append(nn.BatchNorm1d(nhid))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            ninput = nhid
        if nlayers == 0:
            nhid = ninput
        layers.append(nn.Linear(nhid, noutput))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x:   FloatTensor B*ninput
        :return:    FloatTensor B*nouput
        """
        return self.mlp(x)

class FactorizationMachine(nn.Module):

    def __init__(self, reduce_dim=True):
        super().__init__()
        self.reduce_dim = reduce_dim

    def forward(self, x):
        """
        :param x:   FloatTensor B*F*E
        """
        square_of_sum = torch.sum(x, dim=1)**2                  # B*E
        sum_of_square = torch.sum(x**2, dim=1)                  # B*E
        fm = square_of_sum - sum_of_square                      # B*E
        if self.reduce_dim:
            fm = torch.sum(fm, dim=1)                           # B
        return 0.5 * fm                                         # B*E/B