from rmad import *
def main():
    f = lambda x, y: exp(-(sin(x)-cos(y))**Variable(2))
    x = Variable(1.0)
    y = Variable(1.0)
    z = f(x, y)
    gradients = getGradients(z)

    print("The partial derivative of z with respect to x =", x.grad)
    print("The partial derivative of z with respect to y =", y.grad)

    # finite differences
    # h = 1e-8
    # f = lambda x, y: math.exp(-(math.sin(x)-math.cos(y))**2)
    # dx = (f(1+h,1)-f(1,1))/h
    # dy = (f(1,1+h)-f(1,1))/h
    # print(dx, dy)
if __name__ == '__main__':
    main()