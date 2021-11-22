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
    N = 200
    M = 100
    current_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'Q3_data')
    P = np.zeros((N, N), dtype=np.float32)
    q = np.zeros((N), dtype=np.float32)
    A = np.zeros((M, N), dtype=np.float32)
    b = np.zeros((M), dtype=np.float32)
    P_place = os.path.join(current_dir, 'P.csv')
    q_place = os.path.join(current_dir, 'q.csv')
    P_place = os.path.join(current_dir, 'P.csv')
    q_place = os.path.join(current_dir, 'q.csv')
    with open(current_place, 'r') as csvfile:
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
        x [numpy float array], [50 or 100]: [the best x data]
    """ 
    current_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result')
    if not os.path.exists(current_dir):
        os.mkdir(current_dir)
    current_dir = os.path.join(current_dir, 'Q2_data')
    if not os.path.exists(current_dir):
        os.mkdir(current_dir)
    save_place = os.path.join(current_dir, 'A_' + str(size) + '.csv')
    with open(save_place, 'w') as csvfile:
        data_writer = csv.writer(csvfile)
        data_writer.writerow(x)


def f(P, q, x):
    """Get f(x)

    Args:
        P [numpy float array], [N * N]: [P]
        q [numpy float array], [N]: [q]
        x [numpy float array], [N]: [x]
    
    Return:
        fx [float]: [fx]
    """
    fx = np.matmul(x.T, np.matmul(P, x)) / 2 + np.matmul(q.T, x)
    return fx 

def df(P, q, x):
    """Get df(x)

    Args:
        P [numpy float array], [N * N]: [P]
        q [numpy float array], [N]: [q]
        x [numpy float array], [N]: [x]
    
    Return:
        dfx [numpy float array], [N]: [dfx]
    """
    dfx = np.matmul(P, x) + q 
    return dfx 

def d2f(P, q, x):
    """Get d2f(x)

    Args:
        P [numpy float array], [N * N]: [P]
        q [numpy float array], [N]: [q]
        x [numpy float array], [N]: [x]
    
    Return:
        d2fx [numpy float array], [N * N]: [dfx]
    """
    return P 

def get_newton_direction(x, dfx, d2fx, A, b):
    """Get newton descent direction

    Args:
        x [numpy float array], [N]: [x]
        dfx [numpy float array], [N]: [dfx]
        d2fx [numpy float array], [N * N]: [dfx]
        A [numpy float array], [M * N]: [A]
        b [numpy float array], [M]: [b]
    
    Returns:
        newton_direction [numpy float array], [N]: [the newton descent direction]
    """
    M = b.shape[0]
    N = x.shape[0]
    zeros = np.zeros((M, M), dtype=np.float32)
    left_matrix_left = np.concatenate((d2fx, A), dim=0) #(M + N) * N
    left_matrix_right = np.concatenate((A.T, zeros), dim=0) #(M + N) * M 
    left_matrix = np.concatenate((left_matrix_left, left_matrix_right), dim=1) #(M + N) * (M + N)
    zeros = np.zeros((M), dtype=np.float32)
    right_matrix = np.concatenate((-dfx, zeros), dim=0) #(M + N)
    direction = np.matmul(np.linalg.inv(left_matrix), right_matrix)
    newton_direction = direction[0:N]
    return newton_direction

def get_u_best(P, q, A, b, best_x):
    """Get the best u, which is the best result of the dual problem

    Args:
        P [numpy float array], [N * N]: [P]
        q [numpy float array], [N]: [q]
        A [numpy float array], [M * N]: [A]
        b [numpy float array], [M]: [b]
        best x [numpy float array], [N]: [the best x of the original problem]
  
    Returns:
        best_u [numpy float array], [M]: [best u of the dual problem]
    """
    a1 = -np.linalg(np.matmul(A, A.T))
    a2 = np.matmul(A, np.matmul(P, best_x) + q)
    best_u = np.matmul(a1, a2)
    return best_u

def newton_method(P, q, A, b, x0, alpha, beta, epsilon):
    """Use newton method to optimize

    Args:
        P [numpy float array], [N * N]: [P]
        q [numpy float array], [N]: [q]
        A [numpy float array], [M * N]: [A]
        b [numpy float array], [M]: [b]
        x0 [numpy float array], [N]: [starting point]
        alpha [float]: [the parameter in backtracking]
        beta [float]: [the parameter in backtracking]
        epsilon [float]: [the stopping criterion]
    
    Returns:
        x_best [numpy float array], [N]: [the best x]
        u_best [numpy float array], [M]
        y_best [float]: [the best y]
        log_dist_list [float array]: [the log distance between predicted y and y*]
        t_list [float array]: [the t values]
    """
    t_list = []
    y_list = []
    log_dist_list = []
    x = x0 
    extra_steps = 0
    while True:
        #calculate directions
        fx = f(P, q, x)
        dfx = df(P, q, x)
        d2fx = d2f(P, q, x)
        nt_direction = get_newton_direction(x, dfx, d2fx, A, b)
        lambda_2 = np.matmul(nt_direction.T, np.matmul(d2fx, nt_direction))

        #calculate t
        t = 1.0
        while True:
            dist = f(P, q, x + t * nt_direction) - fx - alpha * t * np.matmul(df(P, q, x).T, nt_direction)
            if dist <= 0:
                break 
            t = beta * t
      
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
                u_best = get_u_best(P, q, A, b, x_best)
            extra_steps += 1
            if extra_steps >= 3:
                p_star = fx
                break             
        x = x + t * nt_direction
        
    for i in range(len(y_list)):
        log_dist = log(abs(y_list[i] - p_star))
        log_dist_list.append(log_dist)
    
    return x_best, u_best, y_best, log_dist_list, t_list  
    

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
    plt.scatter(k, log_d, color='b', zorder=1)
    plt.plot(k, log_d, color='b', zorder=1)
    plt.show()
    
    plt.xlabel('k')
    plt.ylabel('tk')
    plt.scatter(k, t, color='b', zorder=1)
    plt.plot(k, t, color='b', zorder=1)
    plt.show()
    
def main():
    P, q, A, b = load_data()
    #TODO x0
    alpha = 0.4
    beta = 0.5
    epsilon = 1e-5 
    x_best, u_best, y_best, log_dist_list, t_list = newton_method(P, q, A, b, x0, alpha, beta, epsilon)
    visualize(log_dist_list, t_list)
    save_data(x_best, u_best)
    print('y best:', y_best)

if __name__ == "__main__":
    main()