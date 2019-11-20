from typing import Tuple

import torch
from torch.nn import Linear
from torch.nn.functional import leaky_relu


class DistributedMLP(torch.nn.Module):

    def __init__(self, in_num: int, hidden_layers: Tuple[int, int], out_num: int):
        # nn.Linear initialises parameters with kaiming uniform
        super(DistributedMLP, self).__init__()
        self.linear1 = Linear(in_num, hidden_layers[0])
        self.linear2 = Linear(hidden_layers[0], hidden_layers[1])
        self.linear3 = Linear(hidden_layers[1], out_num)

    def forward(self, x):
        x = leaky_relu(self.linear1(x))
        x = leaky_relu(self.linear2(x))
        return self.linear3(x)
