import torch
import torch.multiprocessing as mp
from torch import nn
from torch.optim import SGD
from torchvision import datasets, transforms

from distributed_model.model import DistributedMLP
from distributed_model.train import train, test


# Using the example from https://github.com/pytorch/examples/blob/master/mnist_hogwild/train.py as a base


if __name__ == '__main__':
    # This code currently runs with two processes on my mac (cpu) have yet to test it across two gpus
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}
    mp.set_start_method('spawn')

    model = DistributedMLP(28 * 28, (512, 512), 10).to(device)
    model.share_memory()

    mnist_transforms = [
        transforms.ToTensor()
        ,transforms.Normalize((0.1307,), (0.3081,))
        ]

    data_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose(mnist_transforms)
                       ),
        batch_size=10, shuffle=True, num_workers=1,
        **dataloader_kwargs)

    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()

    processes = []
    for i in range(4):
        p = mp.Process(target=train, args=(model, 5, data_loader, optimizer, loss_function, True))
        p.start()
        processes.append(p)

    for p in processes: p.join()

    test_data_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=10, shuffle=True, num_workers=1,
        **dataloader_kwargs)

    test(model, test_data_loader, loss_function, resize=True)
