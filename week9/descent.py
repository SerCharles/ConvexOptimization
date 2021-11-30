import numpy as np
import matplotlib.pyplot as plt

def f(A, x):
    return float(np.matmul(np.matmul(x.T, A), x) / 2)

def df(A, x):
    return np.matmul(A, x)

def descent(A, x0, t, delta):
    """Constant step gradient descent

    Args:
        A [numpy float array]: [the coeffiecient]
        x0 [numpy float array]: [the starting point]
        t [float]: [the learning rate]
        delta [float]: [the minimal error]
        
    Returns:
        x_list [array of numpy float array]: [the trace of x]
        y_list [array of float]: [the trace of y]
    """
    x_list = []
    y_list = []
    x = x0 
    while True:
        y = f(A, x)
        x_list.append(x)
        y_list.append(y)
        gradient = df(A, x)
        gradient_norm = np.linalg.norm(gradient, ord=2)
        if gradient_norm < delta: 
            break 
        x = x - t * gradient
    print(len(y_list), x, y)
    return x_list, y_list
    
def descent_backtrack(A, x0, alpha, beta, delta):
    """Constant step gradient descent

    Args:
        A [numpy float array]: [the coeffiecient]
        x0 [numpy float array]: [the starting point]
        alpha [float]: [the parameter in backtracking]
        beta [float]: [the parameter in backtracking]
        delta [float]: [the minimal error]
        
    Returns:
        x_list [array of numpy float array]: [the trace of x]
        y_list [array of float]: [the trace of y]
    """
    x_list = []
    y_list = []
    x = x0 
    while True:
        y = f(A, x)
        x_list.append(x)
        y_list.append(y)
        gradient = df(A, x)
        gradient_norm = np.linalg.norm(gradient, ord=2)
        if gradient_norm < delta: 
            break 
        
        #calculate t
        t = 1.0
        while True:
            dist = f(A, x - t * gradient) - y + alpha * t * float(gradient_norm) * float(gradient_norm)
            if dist <= 0:
                break 
            t = beta * t
        
        x = x - t * gradient
    print(len(y_list), x, y)
    return x_list, y_list


def descent_exact(A, x0, delta):
    """Constant step gradient descent

    Args:
        A [numpy float array]: [the coeffiecient]
        x0 [numpy float array]: [the starting point]
        delta [float]: [the minimal error]
        
    Returns:
        x_list [array of numpy float array]: [the trace of x]
        y_list [array of float]: [the trace of y]
    """
    x_list = []
    y_list = []
    x = x0 
    while True:
        y = f(A, x)
        x_list.append(x)
        y_list.append(y)
        gradient = df(A, x)
        gradient_norm = np.linalg.norm(gradient, ord=2)
        if gradient_norm < delta: 
            break 
        
        #get t
        a2 = np.matmul(A, A)
        a3 = np.matmul(a2, A)
        up = float(np.matmul(x.T, np.matmul(a2, x)))
        down = float(np.matmul(x.T, np.matmul(a3, x)))
        t = up / down
        
        x = x - t * gradient
    print(len(y_list), x, y)
    return x_list, y_list

def visualize(x_list, y_list):
    """Visualize the x1-x2 and k-f(xk) plots

    Args:
        x_list [array of numpy float array]: [the trace of x]
        y_list [array of float]: [the trace of y]
    """
    n = len(y_list)
    x = np.stack(x_list, axis=0)
    x1 = x[:, 0]
    x2 = x[:, 1]
    k = np.arange(0, n)
    y = np.array(y_list)
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('The places of x in the 2D plane')
    plt.scatter(x1, x2)
    plt.show()
    
    plt.xlabel('k')
    plt.ylabel('f(xk)')
    plt.title('The change of function values')
    plt.semilogy(k, y)
    plt.show()

def main():
    A = np.array([[1.0, 0.0], [0.0, 100.0]], dtype=np.float32)
    x0 = np.array([100.0, 1.0], dtype=np.float32)
    t = 2 / 101
    alpha = 0.4
    beta = 0.5
    delta = 1e-8
    x_1, y_1 = descent(A, x0, t, delta)
    visualize(x_1, y_1)
    x_2, y_2 = descent_backtrack(A, x0, alpha, beta, delta)
    visualize(x_2, y_2)
    x_3, y_3 = descent_exact(A, x0, delta)
    visualize(x_3, y_3)

if __name__ == "__main__":
    main()
