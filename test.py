from rmad import *
def main():
    f = lambda x, y: Tensor.Primitives.exp(-(Tensor.Primitives.sin(x)-Tensor.Primitives.cos(y))**2)
    x = Tensor(1.0)
    y = Tensor(1.0)
    z = f(x, y)
    calcGradients(z)
    print("The partial derivative of z with respect to x =", x.grad)
    print("The partial derivative of z with respect to y =", y.grad)

    # finite differences
    # h = 1e-9
    # f = lambda x, y: math.exp(-(math.sin(x)-math.cos(y))**2)
    # dx = (f(1+h,1)-f(1,1))/h
    # dy = (f(1,1+h)-f(1,1))/h
    # print(dx, dy)
if __name__ == '__main__':
    main()