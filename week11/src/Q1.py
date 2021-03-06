import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """Get f(x)

    Args:
        x [numpy double array], [2]: [x = (x1, x2)]
        
    Returns:
        fx [double]: [f(x)]
    """
    x0 = x[0]
    x1 = x[1]
    fx = (x0 * x0 + 100 * x1 * x1) / 2
    return fx 

def df(x):
    """Get df(x)

    Args:
        x [numpy double array], [2]: [x = (x1, x2)]
        
    Returns:
        dfx [numpy double array], [2]: [df(x)]
    """
    dfx = np.zeros((2), dtype=np.float64)
    dfx[0] = x[0]
    dfx[1] = x[1] * 100
    return dfx 
    

def descent(x0, alpha, beta, stop_criterion):
    """Constant step gradient descent

    Args:
        x0 [numpy double array], [2]: [the starting point]
        alpha [double]: [the alpha parameter of newton method]
        beta [double]: [the beta parameter of newton method]
        stop_criterion [double]: [the minimal error]
        
    Returns:
        best_x [numpy double array], [2]: [the best x]
        best_y [double]: [the best y]
        x_list [array of numpy double array]: [the trace of x]
        y_list [array of double]: [the trace of y]
    """
    x_list = []
    y_list = []
    x = x0 
    while True:
        y = f(x)
        x_list.append(x)
        y_list.append(y)
        gradient = df(x)
        gradient_norm = np.linalg.norm(gradient, ord=2)
        if gradient_norm < stop_criterion: 
            best_x = x 
            best_y = y
            break 
        
        learning_rate = 1.0
        while True:
            dist = f(x - learning_rate * gradient) - y + alpha * learning_rate * float(gradient_norm) * float(gradient_norm)
            if dist <= 0:
                break 
            learning_rate = beta * learning_rate
        
        x = x - gradient * learning_rate
    return best_x, best_y, x_list, y_list

def descent_momentum(x0, learning_rate, momentum_rate, stop_criterion):
    """Constant step gradient descent with momentum

    Args:
        x0 [numpy double array], [2]: [the starting point]
        learning_rate [double]: [the learning rate]
        momentum_rate [double]: [the momentum_rate]
        stop_criterion [double]: [the minimal error]
        
    Returns:
        best_x [numpy double array], [2]: [the best x]
        best_y [double]: [the best y]
        x_list [array of numpy double array]: [the trace of x]
        y_list [array of double]: [the trace of y]
    """
    x_list = []
    y_list = []
    x = x0 
    momentum = np.zeros((2), dtype=np.float64)
    while True: 
        y = f(x)
        x_list.append(x)
        y_list.append(y)
        gradient = df(x)
        gradient_norm = np.linalg.norm(gradient, ord=2)
        if gradient_norm < stop_criterion: 
            best_x = x 
            best_y = y
            break 
        old_x = x
        x = x - gradient * learning_rate + momentum * momentum_rate
        momentum = x - old_x 
    return best_x, best_y, x_list, y_list
                

def visualize(x_list_1, y_list_1, x_list_2, y_list_2):
    """Compare the k-f(xk) plot between gradient descent and gradient descent with momentum

    Args:
        x_list_1 [array of numpy float array]: [the trace of x of gradient descent]
        y_list_1 [array of float]: [the trace of y of gradient descent]
        x_list_2 [array of numpy float array]: [the trace of x of gradient descent with momentum]
        y_list_2 [array of float]: [the trace of y of gradient descent with momentum]
    """
    x1 = np.array(x_list_1)
    x2 = np.array(x_list_2)
    y1 = np.array(y_list_1)
    y2 = np.array(y_list_2)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('The places of x in the 2D plane')
    plt.scatter(x1[:, 0], x1[:, 1], color='red', zorder=1)
    plt.plot(x1[:, 0], x1[:, 1], color='red', zorder=2, label='Gradient Descent')
    plt.scatter(x2[:, 0], x2[:, 1], color='blue', zorder=1)
    plt.plot(x2[:, 0], x2[:, 1], color='blue', zorder=2, label='Gradient Descent with Momentum')
    plt.legend()
    plt.show()


    plt.xlabel('k')
    plt.ylabel('f(xk)')
    plt.title('The change of function values')
    plt.semilogy(y1, label='Gradient Descent', color='red')
    plt.semilogy(y2, label='Gradient Descent with Momentum', color='blue')
    plt.legend()
    plt.show()

def main():
    x0 = np.array([100.0, 1.0], dtype=np.float64)
    alpha = 4 / 121
    beta = 81 / 121 
    delta = 1e-8
    best_x1, best_y1, x1, y1 = descent(x0, alpha, beta, delta)
    print('The best result of gradient descent: x = ({}, {}), y = {}'.format(best_x1[0], best_x1[1], best_y1))
    best_x2, best_y2, x2, y2 = descent_momentum(x0, alpha, beta, delta)
    print('The best result of gradient descent with momentum: x = ({}, {}), y = {}'.format(best_x2[0], best_x2[1], best_y2))
    visualize(x1, y1, x2, y2)
     


if __name__ == "__main__":
    main()
