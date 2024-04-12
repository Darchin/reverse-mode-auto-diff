from rmad import *
def main():
    f = lambda x, y: exp(-(sin(x)-cos(y))**Variable(2))
    x = Variable(1.0)
    y = Variable(1.0)
    z = f(x, y)
    gradients = getGradients(z)

    print("The partial derivative of z with respect to x =", gradients[x])
    print("The partial derivative of z with respect to y =", gradients[y])

if __name__ == '__main__':
    main()