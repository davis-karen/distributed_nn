import torch
import torch.multiprocessing as mp

from torch import nn
from torch.optim import SGD
from train import train, test


def distribute_training(model, no_of_processes, no_of_epochs, data_loader, test_data_loader):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    mp.set_start_method('spawn')

    model = model.to(device)
    model.share_memory()

    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()

    processes = []
    for i in range(no_of_processes):
        p = mp.Process(target=train, args=(model,no_of_epochs, data_loader, optimizer, loss_function, True))
        p.start()
        processes.append(p)

    for p in processes: p.join()

    test(model, test_data_loader, loss_function, resize=True)
