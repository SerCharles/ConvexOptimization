from math import *
import numpy as np
import matplotlib.pyplot as plt 

def f(x):
    """Get f(x)

    Args:
        x [numpy float array], [2]: [x = (x1, x2)]
        
    Returns:
        fx [float]: [f(x)]
    """
    fx = (10 * x[0] * x[0] + x[1] * x[1]) / 2 + 5 * log(1 + exp(-x[0]-x[1]))
    return fx 

def df(x):
    """Get df(x)

    Args:
        x [numpy float array], [2]: [x = (x1, x2)]
        
    Returns:
        dfx [numpy float array], [2]: [df(x)]
    """
    dfx = np.zeros((2), dtype=np.float)
    right = -5 * exp(-x[0]-x[1]) / (1 + exp(-x[0]-x[1]))
    dfx[0] = 10 * x[0] + right 
    dfx[1] = x[1] + right
    return dfx

def d2f(x):
    """Get d2f(x)

    Args:
        x [numpy float array], [2]: [x = (x1, x2)]
        
    Returns:
        d2fx [numpy float array], [2 * 2]: [d2f(x)]
    """
    d2fx = np.zeros((2, 2), dtype=np.float)
    right = 5 * exp(-x[0]-x[1]) / ((1 + exp(-x[0]-x[1])) ** 2)
    d2fx[0, 0] = 10 + right
    d2fx[0, 1] = right 
    d2fx[1, 0] = right 
    d2fx[1, 1] = 1 + right 
    return d2fx

def newton_method(x0, alpha, beta, epsilon):
    """Constant step gradient descent

    Args:
        x0 [numpy float array]: [the starting point]
        alpha [float]: [the parameter in backtracking]
        beta [float]: [the parameter in backtracking]
        epsilon [float]: [the minimal error]
        
    Returns:
        x_list [array of numpy float array]: [the trace of x]
        log_y_list [array of float]: [the trace of log y]
    """
    x_list = []
    log_y_list = []
    x = x0 
    while True:
        fx = f(x)
        x_list.append(x)
        log_y_list.append(log(fx))
        dfx = df(x)
        d2fx = d2f(x)
        nt_direction = -np.matmul(np.linalg.inv(d2fx), dfx)
        if np.linalg.norm(dfx, ord=2) <= epsilon: 
            break 
        
        #calculate t
        t = 1.0
        while True:
            dist = f(x + t * nt_direction) - fx - alpha * t * np.matmul(df(x).T, nt_direction)
            if dist <= 0:
                break 
            t = beta * t
        
        x = x + t * nt_direction
    print(len(log_y_list), x, fx)
    return x_list, log_y_list


def visualize(x_list, log_y_list):
    """Visualize the x1-x2 and k-f(xk) plots

    Args:
        x_list [array of numpy float array]: [the trace of x]
        log_y_list [array of float]: [the trace of log y]
    """
    n = len(log_y_list)
    x = np.stack(x_list, axis=0)
    x1 = x[:, 0]
    x2 = x[:, 1]
    k = np.arange(0, n)
    y = np.array(log_y_list)
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('The places of x in the 2D plane')
    plt.scatter(x1, x2, color='b', zorder=1)
    plt.plot(x1, x2, color='b', zorder=2)
    plt.show()
    
    plt.xlabel('k')
    plt.ylabel('log(f(xk))')
    plt.title('The change of function values')
    plt.scatter(k, y, color='b', zorder=1)
    plt.plot(k, y, color='b', zorder=2)    
    plt.show()

def main():
    x0 = np.array([0.0, 0.0])
    alpha = 0.4
    beta = 0.5
    epsilon = 1e-8 
    x_list, log_y_list = newton_method(x0, alpha, beta, epsilon)
    visualize(x_list, log_y_list)
    
if __name__ == "__main__":
    main()