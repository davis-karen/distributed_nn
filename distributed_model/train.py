import torch
from torch.nn import Module
from torch.nn.functional import nll_loss, log_softmax


def train(model: Module, epochs: int, data_loader, optimiser, loss_function):
    model.train() # letting pytorch know this is a training pass
    for epoch in range(1, epochs + 1):
        for batch_id, (X, y) in enumerate(data_loader):
            optimiser.zero_grad()# clear previous gradients
            output = model(X)
            loss = loss_function(output, y)
            loss.backward() # computes gradients of all variables with regard to the loss function
            optimiser.step() # applys the gradients to the weights
            if epoch % 10 == 0:
                print(f'Epoch {epoch} Batch : {batch_id} Loss: {loss.item()}')


def test(model: Module, data_loader, loss_function):
    model.eval() #letting pytorch know this is evaluation phase
    test_loss = 0
    accuracy = 0
    with torch.no_grad(): # don't calculate gradients
        for X, y in data_loader:
            print('hi')
            output = model(X)
            test_loss += loss_function(output, y).item()
            pred = output.max(1)[1]
            accuracy += pred.eq(y).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy /= len(data_loader.dataset)
    print(f'Test set: Average loss: {test_loss} | Accuracy: {accuracy}')
    return test_loss, accuracy