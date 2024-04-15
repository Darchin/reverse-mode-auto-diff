import math
from collections import deque

class Tensor:
    def __init__(self, value, children=set()):
        self.value = value
        self.grad = 0
        self.children: set = children
        self._backward = lambda : None
    
    def __str__(self):
        return str(self.value)
    def __repr__(self):
        return self.__str__()
    
    class Primitives:
        def add(var1, var2):
            out = Tensor(value=var1.value + var2.value, children=set([var1, var2]))
            def backward():
                var1.grad = 1 * out.grad
                var2.grad = 1 * out.grad
            out._backward = backward
            return out

        def mul(var1, var2):
            out = Tensor(value=var1.value * var2.value, children=set([var1, var2]))
            def backward():
                var1.grad = var2.value * out.grad
                var2.grad = var1.value * out.grad
            out._backward = backward
            return out

        def neg(var):
            out = Tensor(value=-var.value, children=set([var]))
            def backward():
                var.grad = -1 * out.grad
            out._backward = backward
            return out

        def inv(var):
            out = Tensor(value=1/var.value, children=set([var]))
            def backward():
                var.grad = (-1 / (var.value**2)) * out.grad
            out._backward = backward
            return out

        def pow(var1, var2):
            out = Tensor(value=var1.value ** var2.value, children=set([var1, var2]))
            def backward():
                var1.grad = var2.value*var1.value**(var2.value-1) * out.grad
                var2.grad = var1.value**var2.value*(math.log(var1.value)) * out.grad
            out._backward = backward
            return out

        def sin(var):
            out = Tensor(value=math.sin(var.value), children=set([var]))
            def backward():
                var.grad = math.cos(var.value) * out.grad
            out._backward = backward
            return out

        def cos(var):
            out = Tensor(value=math.cos(var.value), children=set([var]))
            def backward():
                var.grad = -math.sin(var.value) * out.grad
            out._backward = backward
            return out

        def exp(var):
            out = Tensor(value=math.exp(var.value), children=set([var]))
            def backward():
                var.grad = math.exp(var.value) * out.grad
            out._backward = backward
            return out

        def log(var): # Base 'e'
            out = Tensor(value=math.log(var.value), children=set([var]))
            def backward():
                var.grad = 1 / var.value * out.grad
            out._backward = backward
            return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor.Primitives.add(self, other)
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor.Primitives.add(self, Tensor.Primitives.neg(other))
    def __neg__(self):
        return Tensor.Primitives.neg(self)
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor.Primitives.mul(self, other)
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor.Primitives.mul(self, Tensor.Primitives.inv(other))
    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor.Primitives.pow(self, other)
    
def topoSort(root: Tensor) -> deque:
    # Returns a deque with the terminal nodes (ie the inputs x_1,..., x_n) appended first.
    sorted_deque = deque()
    visited = set()
    def ts(v: Tensor):
        if v not in visited:
            visited.add(v)
        for c in v.children:
          ts(c)
        sorted_deque.append(v)
    ts(root)
    return sorted_deque

def calcGradients(func: Tensor):
    "Returns the gradient of the function with respect to its inputs."
    stack = topoSort(func)
    func.grad = 1
    while len(stack) != 0: 
        tensor = stack.pop()
        tensor._backward()