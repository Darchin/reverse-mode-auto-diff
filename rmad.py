import math
from collections import defaultdict

class Variable:
    def __init__(self, value, local_gradients=[]):
        self.value = value
        self.local_gradients = local_gradients
    
    def __add__(self, other):
        # if isinstance(self, Variable) and isinstance(other, Variable):
        return add(self, other)
        # elif not isinstance(other, Variable):
        #     return add(self, Variable(other))
        # elif not isinstance(self, Variable): 
        #     return add(Variable(self), other)
        # else: 
        #     return add(Variable(self), Variable(other))
        
    def __sub__(self, other):
        return add(self, neg(other))
    def __neg__(self):
        return neg(self)
    def __mul__(self, other):
        return mul(self, other)
    def __truediv__(self, other):
        return mul(self, inv(other))
    def __pow__(self, other):
        return pow(self, other)
    
def add(var1: Variable, var2: Variable) -> tuple:
    value = var1.value + var2.value
    local_gradients = [
        (var1, 1),
        (var2, 1)
    ]
    return Variable(value, local_gradients)

def mul(var1: Variable, var2: Variable) -> tuple:
    value = var1.value * var2.value
    local_gradients = [
        (var1, var2.value),
        (var2, var1.value)
    ]
    return Variable(value, local_gradients)

def neg(var: Variable) -> tuple:
    value = -1 * var.value
    local_gradients = [
        (var, -1)
    ]
    return Variable(value, local_gradients)

def inv(var: Variable) -> tuple:
    value = 1 / var.value
    local_gradients = [
        (var, -1 / var.value**2)
    ]
    return Variable(value, local_gradients)

def pow(var1: Variable, var2: Variable) -> tuple:
    value = var1.value ** var2.value
    local_gradients = [
        (var1, var2.value*var1.value**(var2.value-1)),
        (var2, var1.value**var2.value*(math.log(var1.value)))
    ]
    return Variable(value, local_gradients)

def sin(var: Variable) -> tuple:
    value = math.sin(var.value)
    local_gradients = [
        (var, math.cos(var.value))
    ]
    return Variable(value, local_gradients)

def cos(var: Variable) -> tuple:
    value = math.cos(var.value)
    local_gradients = [
        (var, -math.sin(var.value))
    ]
    return Variable(value, local_gradients)

def exp(var: Variable) -> tuple:
    value = math.exp(var.value)
    local_gradients = [
        (var, math.exp(var.value))
    ]
    return Variable(value, local_gradients)

def log(var: Variable) -> tuple: # Base 'e'
    value = math.log(var.value)
    local_gradients = [
        (var, 1/var.value)
    ]
    return Variable(value, local_gradients)

def getGradients(func: Variable) -> dict:
    gradients = defaultdict(lambda: 0)
    "Returns the gradient of the function with respect to its inputs."
    def reversePass(var: Variable, dvar: float):
        for child_var, local_grad in var.local_gradients:
            child_partial = dvar * local_grad
            gradients[child_var] += child_partial
            reversePass(child_var, child_partial)
    reversePass(func, 1)
    return gradients
    
# def main():
#     f = lambda a, b: (a / b - a) * (b / a + a + b) * (a - b)
#     a = Variable(230.3)
#     b = Variable(33.2)
#     y = f(a, b)

#     gradients = autoDiff.getGradients(y)

#     print("The partial derivative of y with respect to a =", gradients[a])
#     print("The partial derivative of y with respect to b =", gradients[b])

# if __name__ == '__main__':
#     main()