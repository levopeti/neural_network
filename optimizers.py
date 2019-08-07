"""
Optimization methods.
http://cs231n.github.io/neural-networks-3/
https://www.quora.com/What-are-differences-between-update-rules-like-AdaDelta-RMSProp-AdaGrad-and-AdaM
"""

import numpy as np


class SGD(object):
    """
    SGD with momentum and nesterov
    It needs learning rate = 1.
    """
    def __init__(self, gamma=0.9, nesterov=False):
        self.name = "SGD"
        # gamma for momentum
        self.gamma = gamma
        self.nesterov = nesterov

    def init(self, W_shape, b_shape):
        velocity_W = np.zeros(W_shape)
        velocity_b = np.zeros(b_shape)

        return velocity_W, velocity_b

    def run(self, W, velocity_W, delta_W, b, velocity_b, delta_b, learning_rate):
        """Cache is velocity for SGD."""

        if learning_rate is None:
            learning_rate = 1

        # update velocity
        velocity_W = np.add((self.gamma * velocity_W), (delta_W * learning_rate))
        velocity_b = np.add((self.gamma * velocity_b), (delta_b * learning_rate))

        # update weights
        W = np.subtract(W, velocity_W)
        b = np.subtract(b, velocity_b)

        return W, b, velocity_W, velocity_b


class Adagrad(object):
    """
    Adaptive learning rate method.
    https://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
    It needs a smaller learning rate (0.1).
    """
    def __init__(self, eps=1e-6):
        self.name = "Adagrad"
        self.eps = eps

    @staticmethod
    def init(*args):
        return None, None

    def run(self, W, cache_W, delta_W, b, cache_b, delta_b, learning_rate):
        """Here cache is second (g) order moment."""
        if learning_rate is None:
            learning_rate = 0.1

        if cache_W is None:
            cache_W = np.power(delta_W, 2)
            cache_b = np.power(delta_b, 2)
        else:
            cache_W += np.power(delta_W, 2)
            cache_b += np.power(delta_b, 2)

        # calculate adapt_lr
        adapt_lr_W = learning_rate / (np.sqrt(cache_W) + self.eps)
        adapt_lr_b = learning_rate / (np.sqrt(cache_b) + self.eps)

        # update weights
        W = np.subtract(W, adapt_lr_W * delta_W)
        b = np.subtract(b, adapt_lr_b * delta_b)

        return W, b, cache_W, cache_b


class Adadelta(object):
    """
    It needs a smaller learning rate (0.1).
    """
    def __init__(self, eps=1e-6, autocorr=0.95):
        self.name = "Adadelta"
        self.eps = eps
        self.autocorr = autocorr

    @staticmethod
    def init(*args):
        return None, None

    def run(self, W, cache_W, delta_W, b, cache_b, delta_b, learning_rate):
        """Here cache is second order moment of gradient (g) and second order moment of velocity (x)."""
        if learning_rate is None:
            learning_rate = 1

        if cache_W is None:
            # [g, x, velocity]
            cache_W = [None, None, None]
            cache_b = [None, None, None]

        if cache_W[0] is None:
            # init g
            cache_W[0] = np.zeros(delta_W.shape)
            cache_b[0] = np.zeros(delta_b.shape)

            # init velocity
            cache_W[2] = learning_rate * delta_W
            cache_b[2] = learning_rate * delta_b

            # init x
            cache_W[1] = (1 - self.autocorr) * np.power(cache_W[2], 2)
            cache_b[1] = (1 - self.autocorr) * np.power(cache_b[2], 2)

        # update g
        cache_W[0] = self.autocorr * cache_W[0] + (1 - self.autocorr) * np.power(delta_W, 2)
        cache_b[0] = self.autocorr * cache_b[0] + (1 - self.autocorr) * np.power(delta_b, 2)

        # calculate adapt_lr
        adapt_lr_W = learning_rate * ((np.sqrt(cache_W[1]) + self.eps) / (np.sqrt(cache_W[0]) + self.eps))
        adapt_lr_b = learning_rate * ((np.sqrt(cache_b[1]) + self.eps) / (np.sqrt(cache_b[0]) + self.eps))

        # update velocity
        cache_W[2] = adapt_lr_W * delta_W
        cache_b[2] = adapt_lr_b * delta_b

        # update x
        cache_W[1] = self.autocorr * cache_W[1] + (1 - self.autocorr) * np.power(cache_W[2], 2)
        cache_b[1] = self.autocorr * cache_b[1] + (1 - self.autocorr) * np.power(cache_b[2], 2)

        # update weights
        W = np.subtract(W, adapt_lr_W * delta_W)
        b = np.subtract(b, adapt_lr_b * delta_b)

        return W, b, cache_W, cache_b


class RMSProp(object):
    """
    It needs a smaller learning rate (0.001).
    https://www.quora.com/What-are-differences-between-update-rules-like-AdaDelta-RMSProp-AdaGrad-and-AdaM
    """
    def __init__(self, eps=1e-6, autocorr=0.95, first_order_momentum=False, gamma=0):
        self.name = "RMSProp"
        self.eps = eps
        self.autocorr = autocorr
        self.first_order_momentum = first_order_momentum
        # gamma for momentum
        self.gamma = gamma

    @staticmethod
    def init(*args):
        return None, None

    def run(self, W, cache_W, delta_W, b, cache_b, delta_b, learning_rate):
        """Here cache is first (m) and second (g) order moment and momentum."""
        if learning_rate is None:
            learning_rate = 0.001

        if cache_W is None:
            # [g, m, velocity]
            cache_W = [None, None, None]
            cache_b = [None, None, None]

        # second order moment
        if cache_W[0] is None:
            cache_W[0] = (1 - self.autocorr) * np.power(delta_W, 2)
            cache_b[0] = (1 - self.autocorr) * np.power(delta_b, 2)
        else:
            cache_W[0] = self.autocorr * cache_W[0] + (1 - self.autocorr) * np.power(delta_W, 2)
            cache_b[0] = self.autocorr * cache_b[0] + (1 - self.autocorr) * np.power(delta_b, 2)

        # first order moment
        if self.first_order_momentum:
            if cache_W[1] is None:
                cache_W[1] = (1 - self.autocorr) * delta_W
                cache_b[1] = (1 - self.autocorr) * delta_b
            else:
                cache_W[1] = self.autocorr * cache_W[1] + (1 - self.autocorr) * delta_W
                cache_b[1] = self.autocorr * cache_b[1] + (1 - self.autocorr) * delta_b
        else:
            cache_W[1] = 0
            cache_b[1] = 0

        # calculate adapt_lr
        adapt_lr_W = learning_rate / (np.sqrt(cache_W[0] - np.power(cache_W[1], 2) + self.eps))
        adapt_lr_b = learning_rate / (np.sqrt(cache_b[0] - np.power(cache_b[1], 2) + self.eps))

        # momentum with gamma
        if self.gamma:
            if cache_W[2] is None:
                cache_W[2] = adapt_lr_W * delta_W
                cache_b[2] = adapt_lr_b * delta_b
            else:
                cache_W[2] = np.add((self.gamma * cache_W[2]), (adapt_lr_W * delta_W))
                cache_b[2] = np.add((self.gamma * cache_b[2]), (adapt_lr_b * delta_b))

            # update weights
            W = np.subtract(W, cache_W[2])
            b = np.subtract(b, cache_b[2])
        else:
            # update weights
            W = np.subtract(W, adapt_lr_W * delta_W)
            b = np.subtract(b, adapt_lr_b * delta_b)

        return W, b, cache_W, cache_b


class Adam(object):
    pass


class Adamax(object):
    pass


class Nadam(object):
    pass


# TODO: Second order



