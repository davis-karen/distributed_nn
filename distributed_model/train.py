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
