import os
import csv
from math import *
import numpy as np
import matplotlib.pyplot as plt 

def load_data(size):
    """Load the data

    Args:
        size [int]: [50 or 100]
    
    Returns:
        data [numpy float array], [50 * 50 or 100 * 100]: [the loaded data]
    """
    current_place = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),\
        'data', 'Q2_data', 'A_' + str(size) + '.csv')
    
    data = np.zeros((size, size), dtype=np.float)
    with open(current_place, 'r') as csvfile:
        data_reader = csv.reader(csvfile)
        for row in data_reader:
            i = data_reader.line_num - 1 
            for j in range(len(row)):
                data[i, j] = float(row[j])
    return data 

def f(A, x):
    """Get f(x)

    Args:
        A [numpy float array], [N * N]: [A]
        x [numpy float array], [N]: [x]
        
    Returns:
        fx [float]: [f(x)]
    """
    N = x.shape[0]
    fx = 0.0
    for i in range(N):
        ai = A[:, i] #N
        fx -= log(1 - np.matmul(ai.T, x))
    for i in range(N):
        fx -= log(1 - x[i] * x[i])
    return fx

def df(A, x):
    """Get df(x)

    Args:
        A [numpy float array], [N * N]: [A]
        x [numpy float array], [N]: [x]
        
    Returns:
        dfx [numpy float array], [N]: [df(x)]
    """
    N = x.shape[0]
    dfx = np.zeros((N), dtype=np.float)
    for i in range(N):
        ai = A[:, i] #N
        dfx = dfx + ai / (1 - np.matmul(ai.T, x))
    for i in range(N):
        dfx[i] = dfx[i] + (2 * x[i]) / (1 - x[i] * x[i])
    return dfx

def d2f(A, x):
    """Get d2f(x)

    Args:
        A [numpy float array], [N * N]: [A]
        x [numpy float array], [N]: [x]
        
    Returns:
        d2fx [numpy float array], [N * N]: [d2f(x)]
    """
    N = x.shape[0]
    d2fx = np.zeros((N, N), dtype=np.float)
    for i in range(N):
        ai = A[:, i] #N
        d2fx = d2fx + np.matmul(ai, ai.T) / ((1 - np.matmul(ai.T, x)) ** 2)
    for i in range(N):
        d2fx[i, i] = d2fx[i, i] + 2 * (1 + x[i] ** 2) / ((1 - x[i] ** 2) ** 2)
    return d2fx

def newton_method(A, x0, epsilon):
    """Use newton method to optimize

    Args:
        A [numpy float array], [N * N]: [A]
        x0 [numpy float array], [N]: [starting point]
        epsilon [float]: [the stopping criterion]
    
    Returns:
        best_y [float]: [the best y]
        log_dist_list [float array]: [the log distance between predicted y and y*]
        t_list [float array]: [the t values]
    """
    t_list = []
    y_list = []
    log_dist_list = []
    x = x0 
    extra_steps = 0
    while True:
        fx = f(A, x)
        dfx = df(A, x)
        d2fx = d2f(A, x)
        nt_direction = -np.matmul(np.linalg.inv(d2fx), dfx)
        lambda_2 = float(-np.matmul(dfx.T, nt_direction))
        lambda_1 = sqrt(lambda_2)
        t = 1 / (1 + lambda_1)
        
        
        if lambda_2 >= epsilon: 
            y_list.append(fx)
            t_list.append(t)
        else: 
            if extra_steps == 0:
                y_list.append(fx)
                t_list.append(t)
            extra_steps += 1
            if extra_steps >= 3:
                best_y = fx
                break             
        x = x + t * nt_direction
    for i in range(len(y_list)):
        log_dist = log(abs(y_list[i] - best_y))
        log_dist_list.append(log_dist)
    
    print(best_y)
    return best_y, log_dist_list, t_list

def visualize(log_dist_list, t_list):
    """Visualize the x1-x2 and k-f(xk) plots

    Args:
        log_dist_list [float array]: [the log distance between predicted y and y*]
        t_list [float array]: [the t values]
    """
    n = len(log_dist_list)
    k = np.arange(0, n)
    log_d = np.array(log_dist_list)
    t = np.array(t_list)
    
    plt.xlabel('k')
    plt.ylabel('log(f(xk)-p*)')
    plt.plot(k, log_d)
    plt.show()
    
    plt.xlabel('k')
    plt.ylabel('tk')
    plt.plot(k, t)
    plt.show()
    
def main():
    A1 = load_data(50)
    A2 = load_data(100)
    x01 = np.zeros(50, dtype=np.float)
    x02 = np.zeros(100, dtype=np.float)
    epsilon = 1e-5 
    best_y, log_dist_list, t_list = newton_method(A1, x01, epsilon)
    visualize(log_dist_list, t_list)
    best_y, log_dist_list, t_list = newton_method(A2, x02, epsilon)
    visualize(log_dist_list, t_list)
        
if __name__ == "__main__":
    main()