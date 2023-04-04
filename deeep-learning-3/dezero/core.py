from typing import Any

import numpy as np


class Variable:
    def __init__(self, data: Any):
        self.data = data


class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x: Any) -> Any:
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: Any) -> Any:
        return x**2


class Exp(Function):
    def forward(self, x: Any) -> Any:
        return np.exp(x)


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

print(y.data)
