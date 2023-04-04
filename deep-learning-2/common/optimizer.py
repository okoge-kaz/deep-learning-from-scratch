import numpy as np


class SGD:
    """
    確率的勾配降下法(Stochastic Gradient Descent)
    """

    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def update(self, params: list, grads: list) -> None:
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class Momentum:
    """
    Momentum SGD
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params: list, grads: list) -> None:
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))

        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]
