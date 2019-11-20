from torch import Tensor
import torch

from distributed_model.model import DistributedMLP
from distributed_model.train import train
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import numpy as np


class TestTrain:
    X = torch.from_numpy(
        np.asarray([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1], [1, 0], [0, 1], [1, 0], [1, 1]],
                   np.float32))

    X_test = torch.from_numpy(np.asarray([[1, 1], [0, 1], [0, 0], [1, 1], [1, 0], [1, 1]], np.float32))

    y = torch.from_numpy(np.asarray([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], np.float32)).squeeze(
        1)

    y_1 = torch.from_numpy(np.asarray([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], np.float32)).squeeze \
            (1)

    def test_train_model_to_predict_0(self):
        mlp = DistributedMLP(2, (3, 3), 1)
        # What type is parameters - Tensor? to represent θ

        dataset = TensorDataset(self.X, self.y)  # create your datset
        batch_size = 3
        data_loader = DataLoader(dataset, batch_size)

        optimizer = SGD(mlp.parameters(), lr=0.001, momentum=0.9)
        loss_function = torch.nn.L1Loss()

        train(mlp, 100, data_loader, optimizer, loss_function)

        y_pred = mlp(self.X_test)
        assert y_pred is not None
        for y in y_pred:
            assert y < 1e-2

    def test_train_model_to_predict_1(self):
        mlp = DistributedMLP(2, (3, 3), 1)
        # What type is parameters - Tensor? to represent θ

        dataset = TensorDataset(self.X, self.y_1)  # create your datset
        batch_size = 3
        data_loader = DataLoader(dataset, batch_size)

        optimizer = SGD(mlp.parameters(), lr=0.001, momentum=0.9)
        loss_function = torch.nn.L1Loss()

        train(mlp, 100, data_loader, optimizer, loss_function)

        y_pred = mlp(self.X_test)
        assert y_pred is not None
        for y in y_pred:
            assert 1 - y < 1e-2
