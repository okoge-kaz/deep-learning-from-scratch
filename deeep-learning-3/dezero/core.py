from typing import Any, Optional

import numpy as np


class Variable:
    def __init__(self, data: Any):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")

        self.data: Any = data
        self.grad: Any = None
        self.creator: Optional["Function"] = None

    def set_creator(self, func: "Function") -> None:
        self.creator = func

    def backward(self) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        creator_funcs: list[Optional["Function"]] = [self.creator]
        while creator_funcs != []:
            creator_func: Optional["Function"] = creator_funcs.pop()
            if creator_func is not None:
                # 関数の入出力を取得
                x, y = creator_func.input, creator_func.output
                x.grad = creator_func.backward(y.grad)

                if x.creator is not None:
                    creator_funcs.append(x.creator)


class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)  # y is np.ndarray or float or int
        output = Variable(as_array(y))
        output.set_creator(self)  # 出力変数に生みの親を覚えさせる

        self.input = input
        self.output = output  # 出力も覚える
        return output

    def forward(self, x: Any) -> Any:
        raise NotImplementedError()

    def backward(self, gy: Any) -> Any:
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: Any) -> Any:
        return x**2

    def backward(self, gy: Any) -> Any:
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: Any) -> Any:
        return np.exp(x)

    def backward(self, gy: Any) -> Any:
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)


def as_array(x: Any) -> np.ndarray:
    """Convert to numpy array.

    Args:
        x (Any): np.ndarray or float or int

    Returns:
        np.ndarray
    """
    if np.isscalar(x):
        return np.array(x)
    return x


x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()

print(x.grad)
