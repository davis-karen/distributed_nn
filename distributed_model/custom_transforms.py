import numpy as np
#Using scipy which seems a little slow
#Code used from here: https://github.com/gokriznastic/vision/blob/add-elastic-transform/torchvision/transforms/functional.py

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
