import math
from math import sqrt

import torch

from distributed_model.model import DistributedMLP


class TestDistributedMLP:

    def test_model_is_created_correctly(self):
        mlp = DistributedMLP(10, (5, 5), 2)
        assert mlp is not None

    def test_weights_are_initialised_with_mean_close_to_zero(self):
        mlp = DistributedMLP(10, (5, 5), 2)
        for module in list(mlp._modules.items()):
            mean = module[1].weight.mean()
            print(f'The mean is {mean}')
            assert module[1].weight.mean().abs() < 12e-2

    def test_weights_are_initialised_with_std_kaiming_init(self):
        mlp = DistributedMLP(10, (5, 5), 2)
        for module in list(mlp._modules.items()):
            standard_deviation = module[1].weight.std()
            expected = self.calc_expected_standard_deviation(module[1].weight.size(1))
            print(f'Actual Standard deviation {standard_deviation} | Expected {expected} ')
            assert abs(standard_deviation - expected) < 1e4

    def calc_expected_standard_deviation(self, fan_in: int):
        # This is just really checking that pytorch initialised the weights as per their code - doesn't check
        # if this is the correct thing to do

        # this is the actual expected as per the Kaiming He paper
        # expected = sqrt(2 / fan_in)

        # However pytorch  does this (default is expected leaky relu)
        # negative slope is set to math.sqrt(5)
        negative_slope = math.sqrt(5)
        gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
        return gain / sqrt(fan_in)

    def test_forward_pass(self):
        # Really just testing that the forward pass doesn't blow up
        mlp = DistributedMLP(10, (3, 3), 2)
        x = torch.randn(1, 10)
        result = mlp.forward(x)
        print(f'The result of the forward pass is {result}')
        assert result is not None
