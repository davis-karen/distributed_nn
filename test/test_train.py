from torch import Tensor
import torch

from distributed_model.model import DistributedMLP
from distributed_model.train import train
from distributed_model.train import test
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import numpy as np


class TestTrain:
    X = torch.from_numpy(
        np.asarray([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1], [1, 0], [0, 1], [1, 0], [1, 1]],
                   np.float32))

    X_test = torch.from_numpy(np.asarray([[1, 1], [0, 1], [0, 0], [1, 1], [1, 0], [1, 1]], np.float32))

    y_test_0 = torch.from_numpy(np.asarray([[0], [0], [0], [0], [0], [0]], np.float32))

    y_test_1 = torch.from_numpy(np.asarray([[1], [1], [1], [1], [1], [1]], np.float32))

    y = torch.from_numpy(np.asarray([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], np.float32))\
        .squeeze(1)

    y_1 = torch.from_numpy(np.asarray([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], np.float32))\
        .squeeze(1)

    def test_train_model_to_predict_0(self):
        mlp = DistributedMLP(2, (3, 3), 1)

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

    def test_test_model(self):
        mlp = DistributedMLP(2, (3, 3), 1)
        optimizer = SGD(mlp.parameters(), lr=0.001, momentum=0.9)
        loss_function = torch.nn.L1Loss()
        dataset = TensorDataset(self.X, self.y_1)
        data_loader = DataLoader(dataset)

        test_dataset = TensorDataset(self.X_test, self.y_test_1)
        test_data_loader = DataLoader(test_dataset)

        train(mlp, 100, data_loader, optimizer, loss_function)
        test_loss, accuracy = test(mlp,  test_data_loader, loss_function)
        assert test_loss is not None
        assert accuracy is not None
