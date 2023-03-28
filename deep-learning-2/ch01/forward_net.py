import numpy as np


class Sigmoid:
    def __init__(self) -> None:
        self.params: list = []

    def forward(self, x: np.ndarray | float) -> np.ndarray | float:
        return 1 / (1 + np.exp(-x))


class Affine:
    """
    Affine transformation layer
    全結合層のこと
    """

    def __init__(self, W: np.ndarray, b: np.ndarray) -> None:
        self.params: list = [W, b]

    def forward(self, x: np.ndarray) -> np.ndarray:
        W, b = self.params
        out = np.dot(x, W) + b
        return out
