import os

import torch
from torch.nn import Module


def train(model: Module, epochs: int, data_loader, optimiser, loss_function, resize=False):
    print('Starting to train')
    process_id = os.getpid()
    model.train() # letting pytorch know this is a training pass
    for epoch in range(1, epochs + 1):
        for batch_id, (X, y) in enumerate(data_loader):
            optimiser.zero_grad()# clear previous gradients
            if resize:
                # TODO better way to handle this resize
                X = X.view(-1, 28 * 28)
            output = model(X)
            loss = loss_function(output, y)
            loss.backward() # computes gradients of all variables with regard to the loss function
            optimiser.step() # applies the gradients to the weights
            if batch_id % 200 == 0:
                print(f'Process : {process_id} | Epoch {epoch} Batch : {batch_id} Loss: {loss.item()}')


def test(model: Module, data_loader, loss_function, resize=False):
    model.eval() #letting pytorch know this is evaluation phase
    test_loss = 0
    accuracy = 0
    with torch.no_grad(): # don't calculate gradients
        for X, y in data_loader:
            if resize:
                # TODO better way to handle this resize
                X = X.view(-1, 28 * 28)
            output = model(X)
            test_loss += loss_function(output, y).item()
            pred = output.max(1)[1]
            accuracy += pred.eq(y).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy /= len(data_loader.dataset)
    print(f'Test set: Average loss: {test_loss} | Accuracy: {accuracy}')
    return test_loss, accuracy