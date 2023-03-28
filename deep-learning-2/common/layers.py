from typing import Optional

import numpy as np


class Layer:
    params: list
    grads: list
    x: Optional[np.ndarray]

    def __init__(self) -> None:
        self.params: list = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MatMul(Layer):
    def __init__(self, W: np.ndarray) -> None:
        self.params: list = [W]
        self.grads: list = [np.zeros_like(W)]
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        (W,) = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        (W,) = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)  # type: ignore # dL/dW = x^T * dout
        self.grads[0][...] = dW
        return dx


class Affine(Layer):
    def __init__(self, W: np.ndarray, b: np.ndarray) -> None:
        self.params: list = [W, b]
        self.grads: list = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)  # type: ignore
        db = np.sum(dout, axis=0)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx
