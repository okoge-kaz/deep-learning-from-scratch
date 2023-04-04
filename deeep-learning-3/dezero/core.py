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


x = Variable(np.array(10))
f = Square()
y = f(x)

print(type(y))
print(y.data)
