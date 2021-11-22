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
        data [numpy double array], [50 * 50 or 100 * 100]: [the loaded data]
    """
    current_place = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),\
        'data', 'Q2_data', 'A_' + str(size) + '.csv')
    
    data = np.zeros((size, size), dtype=np.float64)
    with open(current_place, 'r', encoding="utf-8") as csvfile:
        data_reader = csv.reader(csvfile)
        for row in data_reader:
            i = data_reader.line_num - 1 
            for j in range(len(row)):
                data[i, j] = float(row[j])
    return data 

def save_data(size, x):
    """Load the data

    Args:
        size [int]: [50 or 100]
        x [numpy double array], [50 or 100]: [the best x data]
    """ 
    x = x.reshape(size, 1)
    current_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result')
    if not os.path.exists(current_dir):
        os.mkdir(current_dir)
    current_dir = os.path.join(current_dir, 'Q2')
    if not os.path.exists(current_dir):
        os.mkdir(current_dir)
    save_place = os.path.join(current_dir, 'A_' + str(size) + '.csv')
    with open(save_place, 'w', encoding="utf-8", newline="") as csvfile:
        data_writer = csv.writer(csvfile)
        data_writer.writerows(x)

def f(A, x):
    """Get f(x)

    Args:
        A [numpy double array], [N * N]: [A]
        x [numpy double array], [N]: [x]
        
    Returns:
        fx [double]: [f(x)]
    """
    N = x.shape[0]
    fx = 0.0
    for i in range(N):
        ai = A[i] #N
        fx -= log(1 - np.matmul(ai.T, x))
    for i in range(N):
        fx -= log(1 - x[i] ** 2)
    return fx

def df(A, x):
    """Get df(x)

    Args:
        A [numpy double array], [N * N]: [A]
        x [numpy double array], [N]: [x]
        
    Returns:
        dfx [numpy double array], [N]: [df(x)]
    """
    N = x.shape[0]
    dfx = np.zeros((N), dtype=np.float64)
    for i in range(N):
        ai = A[i] #N
        dfx = dfx + ai / (1 - np.matmul(ai.T, x))
    for i in range(N):
        dfx[i] = dfx[i] + (2 * x[i]) / (1 - x[i] ** 2)
    return dfx

def d2f(A, x):
    """Get d2f(x)

    Args:
        A [numpy double array], [N * N]: [A]
        x [numpy double array], [N]: [x]
        
    Returns:
        d2fx [numpy double array], [N * N]: [d2f(x)]
    """
    N = x.shape[0]
    d2fx = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        ai = A[i] #N
        d2fx = d2fx + np.matmul(ai.reshape(N, 1), ai.reshape(1, N)) / ((1 - np.matmul(ai.T, x)) ** 2)
    for i in range(N):
        d2fx[i, i] = d2fx[i, i] + 2 * (1 + x[i] ** 2) / ((1 - x[i] ** 2) ** 2)
    return d2fx

def newton_method(A, x0, epsilon):
    """Use newton method to optimize

    Args:
        A [numpy double array], [N * N]: [A]
        x0 [numpy double array], [N]: [starting point]
        epsilon [double]: [the stopping criterion]
    
    Returns:
        x_best [numpy double array], [N]: [the best x]
        y_best [double]: [the best y]
        log_dist_list [double array]: [the log distance between predicted y and y*]
        t_list [double array]: [the t values]
    """
    t_list = []
    y_list = []
    log_dist_list = []
    x = x0 
    extra_steps = 0
    while True:
        #calculate directions
        fx = f(A, x)
        dfx = df(A, x)
        d2fx = d2f(A, x)
        nt_direction = -np.matmul(np.linalg.inv(d2fx), dfx)
        lambda_2 = float(-np.matmul(dfx.T, nt_direction))
        lambda_1 = sqrt(lambda_2)
        t = 1 / (1 + lambda_1)
      
        #newton descent
        if lambda_2 >= epsilon: 
            y_list.append(fx)
            t_list.append(t)
        else: 
            if extra_steps == 0:
                y_list.append(fx)
                t_list.append(t)
                x_best = x
                y_best = fx
            extra_steps += 1
            if extra_steps >= 3:
                p_star = fx
                break             
        x = x + t * nt_direction
    for i in range(len(y_list)):
        log_dist = log(abs(y_list[i] - p_star))
        log_dist_list.append(log_dist)
    
    return x_best, y_best, log_dist_list, t_list

def visualize(log_dist_list, t_list):
    """Visualize the x1-x2 and k-f(xk) plots

    Args:
        log_dist_list [double array]: [the log distance between predicted y and y*]
        t_list [double array]: [the t values]
    """
    n = len(log_dist_list)
    k = np.arange(0, n)
    log_d = np.array(log_dist_list)
    t = np.array(t_list)
    
    plt.xlabel('k')
    plt.ylabel('log(f(xk)-p*)')
    plt.scatter(k, log_d, color='b', zorder=1)
    plt.plot(k, log_d, color='b', zorder=1)
    plt.show()
    
    plt.xlabel('k')
    plt.ylabel('tk')
    plt.scatter(k, t, color='b', zorder=1)
    plt.plot(k, t, color='b', zorder=1)
    plt.show()
    
def main():
    A1 = load_data(50)
    A2 = load_data(100)
    x01 = np.zeros(50, dtype=np.float64)
    x02 = np.zeros(100, dtype=np.float64)
    epsilon = 1e-5 
    x_best, y_best, log_dist_list, t_list = newton_method(A1, x01, epsilon)
    visualize(log_dist_list, t_list)
    save_data(50, x_best)
    print('y best:', y_best)
    x_best, y_best, log_dist_list, t_list = newton_method(A2, x02, epsilon)
    visualize(log_dist_list, t_list)
    save_data(100, x_best)
    print('y best:', y_best)

if __name__ == "__main__":
    main()