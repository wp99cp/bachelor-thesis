import numpy as np

from configs.config import IMAGE_SIZE


def __heaviside_lambda(x):
    return -np.sign(x) * x + 1


def __heaviside_lambda_2D(x, y):
    return __heaviside_lambda(x) * __heaviside_lambda(y)


def smooth_kern(kernel_size_x=IMAGE_SIZE, kernel_size_y=IMAGE_SIZE):
    """

    :param kernel_size_x:
    :param kernel_size_y:

    :return: a 2D kernel with a smooth transition from 0 to 1

    """

    xs = np.linspace(-1, 1, kernel_size_x)
    ys = np.linspace(-1, 1, kernel_size_y)
    xv, yv = np.meshgrid(xs, ys)

    return __heaviside_lambda_2D(xv, yv)
