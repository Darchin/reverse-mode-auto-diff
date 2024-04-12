import math

class Variable:
    def __init__(self, value, local_gradients=[]):
        self.value: float = value
        self.local_gradients = local_gradients
    
    def __add__(self, other):
        return Primitives.add(self, other)
    def __sub__(self, other):
        return Primitives.add(self, Primitives.neg(other))
    def __mul__(self, other):
        return Primitives.mul(self, other)
    def __truediv__(self, other):
        return Primitives.mul(self, Primitives.inv(other))
    
class Primitives:
    def add(var1: Variable, var2: Variable) -> tuple:
        value = var1.value + var2.value
        local_gradients = (
            (var1, 1),
            (var2, 1),
        )
        return Variable(value, local_gradients)

    def mul(var1: Variable, var2: Variable) -> tuple:
        value = var1.value * var2.value
        local_gradients = (
            (var1, var2.value),
            (var2, var1.value),
        )
        return Variable(value, local_gradients)

    def neg(var: Variable) -> tuple:
        value = -1 * var.value
        local_gradients = (
            (var, -1)
        )
        return Variable(value, local_gradients)

    def inv(var: Variable) -> tuple:
        value = 1 / var.value
        local_gradients = (
            (var, -1 / var.value**2)
        )
        return Variable(value, local_gradients)

    def sin(var: Variable) -> tuple:
        value = math.sin(var.value)
        local_gradients = (
            (var, math.cos(var.value))
        )
        return Variable(value, local_gradients)

    def cos(var: Variable) -> tuple:
        value = math.cos(var.value)
        local_gradients = (
            (var, -math.sin(var.value))
        )
        return Variable(value, local_gradients)

    def exp(var: Variable) -> tuple:
        value = math.exp(var.value)
        local_gradients = (
            (var, math.exp(var.value))
        )
        return Variable(value, local_gradients)

    def log(var: Variable) -> tuple: # Base 'e'
        value = math.log(var.value)
        local_gradients = (
            (var, 1/var.value)
        )
        return Variable(value, local_gradients)
