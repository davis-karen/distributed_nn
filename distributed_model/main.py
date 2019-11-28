import torch
import torch.multiprocessing as mp

from torch import nn
from torch.optim import SGD
from torchvision import datasets, transforms
from torchvision.transforms import RandomAffine

from distributed_model.model import DistributedMLP
from distributed_model.train import train, test
import numpy as np


class ElasticTransform(object):
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    Args:
        alpha (float): Scaling factor as described in [Simard2003] Default value is 100
        sigma (float): Elasticity coefficient as described in [Simard2003] Default value is 10
        random_state (int, optional): Random state to initialize the Gaussian kernel
    """

    def __init__(self, alpha=40, sigma=2, random_state=None):
        if random_state is None:
            self.random_state = np.random.RandomState(None)
        else:
            self.random_state = random_state

        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img):
        from scipy.ndimage.interpolation import map_coordinates
        from scipy.ndimage.filters import gaussian_filter

        img = np.asarray(img)

        if len(img.shape) < 3:
            img = img.reshape(img.shape[0], img.shape[1], -1)

        shape = img.shape

        dx = gaussian_filter((self.random_state.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((self.random_state.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))

        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        distorted_image = map_coordinates(img, indices, order=1, mode='reflect')
        distorted_image = distorted_image.reshape(img.shape)
        return distorted_image


if __name__ == '__main__':
    # This code currently runs with two processes on my mac (cpu) have yet to test it across two gpus
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}
    mp.set_start_method('spawn')

    model = DistributedMLP(28 * 28, (256, 256), 10).to(device)
    model.share_memory()

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
        batch_size=200, shuffle=True, num_workers=1,
        **dataloader_kwargs)

    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()

    processes = []
    for i in range(1):
        p = mp.Process(target=train, args=(model, 5, data_loader, optimizer, loss_function, True))
        p.start()
        processes.append(p)

    for p in processes: p.join()

    test_data_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=200, shuffle=True, num_workers=1,
        **dataloader_kwargs)

    test(model, test_data_loader, loss_function, resize=True)
