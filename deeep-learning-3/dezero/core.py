from typing import Any, Optional

import numpy as np


class Variable:
    def __init__(self, data: Any):
        self.data: Any = data
        self.grad: Any = None
        self.creator: Optional["Function"] = None

    def set_creator(self, func: "Function") -> None:
        self.creator = func


class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
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


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)

print(x.grad)

assert y.creator == C
assert y.creator.input == b  # type: ignore
assert b.creator == B
assert b.creator.input == a  # type: ignore
assert a.creator == A
assert a.creator.input == x  # type: ignore
