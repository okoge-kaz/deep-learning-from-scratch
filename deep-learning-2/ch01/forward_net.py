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


class TwoLayerNet:
    params: list

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE = input_size, hidden_size, output_size

        # Initialize weights and biases
        W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE)
        b1 = np.random.randn(HIDDEN_SIZE)
        W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE)
        b2 = np.random.randn(OUTPUT_SIZE)

        # Create layers
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2),
        ]
        self.params: list = []

        for layer in self.layers:
            self.params += layer.params


# Sigmoid, AffineクラスがLayerクラスを継承しているとすると、+ 演算子などで mypy error がでなくなる。
# 研究のときは、どこまで型を厳密にするかを考える必要がある。
