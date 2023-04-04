from typing import Any


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
