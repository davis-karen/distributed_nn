import logging
import sys

import fire
import torch
from torchvision import datasets, transforms
from model import DistributedMLP
from processing import distribute_training

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Run(object):

    def go(self, batch_size:int=5, no_of_processes:int=1, epochs:int=5):
        logging.info(f'Starting training with batch_size {batch_size} | processes {no_of_processes} | epochs {epochs}')
        use_cuda = torch.cuda.is_available()
        dataloader_kwargs = {'pin_memory': True} if use_cuda else {}
        model = DistributedMLP(28 * 28, (256, 256), 10)

        mnist_transforms = transforms.Compose([
            #      RandomAffine([-15, + 15]),
            #      ElasticTransform(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        data_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=mnist_transforms
                           ),
            batch_size=batch_size, shuffle=True, num_workers=1,
            **dataloader_kwargs)

        test_data_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=batch_size, shuffle=True, num_workers=1,
            **dataloader_kwargs)

        distribute_training(model, no_of_processes, epochs, data_loader, test_data_loader)


if __name__ == '__main__':
    fire.Fire(Run)
