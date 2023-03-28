import numpy as np


class Sigmoid:
    def __init__(self) -> None:
        self.params: list = []

    def forward(self, x: np.ndarray | float) -> np.ndarray | float:
        return 1 / (1 + np.exp(-x))
