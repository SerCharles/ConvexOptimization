import os
import csv
from math import *
import numpy as np
import matplotlib.pyplot as plt 

def load_data():
    """Load the data
    
    Returns:
        P [numpy double array], [N * N]: [P]
        q [numpy double array], [N]: [q]
        A [numpy double array], [M * N]: [A]
        b [numpy double array], [M]: [b]    
    """
    N = 200
    M = 100
    current_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'Q3_data')
    P = np.zeros((N, N), dtype=np.float64)
    q = np.zeros((N, 1), dtype=np.float64)
    A = np.zeros((M, N), dtype=np.float64)
    b = np.zeros((M, 1), dtype=np.float64)
    P_place = os.path.join(current_dir, 'P.csv')
    q_place = os.path.join(current_dir, 'q.csv')
    A_place = os.path.join(current_dir, 'A.csv')
    b_place = os.path.join(current_dir, 'b.csv')
    
    with open(P_place, 'r', encoding="utf-8-sig") as csvfile:
        data_reader = csv.reader(csvfile)
        for row in data_reader:
            i = data_reader.line_num - 1 
            for j in range(len(row)):
                P[i, j] = float(row[j])

    with open(q_place, 'r', encoding="utf-8-sig") as csvfile:
        data_reader = csv.reader(csvfile)
        for row in data_reader:
            i = data_reader.line_num - 1 
            for j in range(len(row)):
                q[i, j] = float(row[j])
    q = q.reshape(N)
    
    with open(A_place, 'r', encoding="utf-8-sig") as csvfile:
        data_reader = csv.reader(csvfile)
        for row in data_reader:
            i = data_reader.line_num - 1 
            for j in range(len(row)):
                A[i, j] = float(row[j])

    with open(b_place, 'r', encoding="utf-8-sig") as csvfile:
        data_reader = csv.reader(csvfile)
        for row in data_reader:
            i = data_reader.line_num - 1 
            for j in range(len(row)):
                b[i, j] = float(row[j])
    b = b.reshape(M)
    return P, q, A, b
    
def save_data(best_x, best_u):
    """Load the data

    Args:
        best_x [numpy double array], [N]: [the best x]
        best_u [numpy double array], [M]: [the best u]
    """ 
    N = best_x.shape[0]
    M = best_u.shape[0]
    best_x = best_x.reshape(N, 1)
    best_u = best_u.reshape(M, 1)
    current_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result')
    if not os.path.exists(current_dir):
        os.mkdir(current_dir)
    current_dir = os.path.join(current_dir, 'Q3')
    if not os.path.exists(current_dir):
        os.mkdir(current_dir)
    
    x_place = os.path.join(current_dir, 'x.csv')
    with open(x_place, 'w', encoding="utf-8", newline="") as csvfile:
        data_writer = csv.writer(csvfile)
        data_writer.writerows(best_x)
    u_place = os.path.join(current_dir, 'u.csv')
    with open(u_place, 'w', encoding="utf-8", newline="") as csvfile:
        data_writer = csv.writer(csvfile)
        data_writer.writerows(best_u)

def get_x0(A, b):
    """Get the x0 that satisfies Ax0 = b

    Args:
        A [numpy double array], [M * N]: [A]
        b [numpy double array], [M]: [b] 
    
    Returns:
        x0 [numpy double array], [N]: [starting point]
    """
    M, N = A.shape
    A_down_left = np.zeros((N - M, M), dtype=np.float64)
    A_down_right = np.identity(N - M, dtype=np.float64)
    A_down = np.concatenate((A_down_left, A_down_right), axis=1) #(N - M) * N
    A_extend = np.concatenate((A, A_down), axis=0) #N * N
    b_down = np.zeros((N - M), dtype=np.float64)
    b_extend = np.concatenate((b, b_down), dtype=np.float64)
    x0 = np.matmul(np.linalg.inv(A_extend), b_extend)
    return x0


def f(P, q, x):
    """Get f(x)

    Args:
        P [numpy double array], [N * N]: [P]
        q [numpy double array], [N]: [q]
        x [numpy double array], [N]: [x]
    
    Return:
        fx [double]: [fx]
    """
    fx = np.matmul(x.T, np.matmul(P, x)) / 2 + np.matmul(q.T, x)
    return fx 

def df(P, q, x):
    """Get df(x)

    Args:
        P [numpy double array], [N * N]: [P]
        q [numpy double array], [N]: [q]
        x [numpy double array], [N]: [x]
    
    Return:
        dfx [numpy double array], [N]: [dfx]
    """
    dfx = np.matmul(P, x) + q 
    return dfx 

def d2f(P, q, x):
    """Get d2f(x)

    Args:
        P [numpy double array], [N * N]: [P]
        q [numpy double array], [N]: [q]
        x [numpy double array], [N]: [x]
    
    Return:
        d2fx [numpy double array], [N * N]: [dfx]
    """
    return P 

def get_newton_direction(x, dfx, d2fx, A, b):
    """Get newton descent direction

    Args:
        x [numpy double array], [N]: [x]
        dfx [numpy double array], [N]: [dfx]
        d2fx [numpy double array], [N * N]: [dfx]
        A [numpy double array], [M * N]: [A]
        b [numpy double array], [M]: [b]
    
    Returns:
        newton_direction [numpy double array], [N]: [the newton descent direction]
    """
    M = b.shape[0]
    N = x.shape[0]
    zeros = np.zeros((M, M), dtype=np.float64)
    left_matrix_left = np.concatenate((d2fx, A), axis=0) #(M + N) * N
    left_matrix_right = np.concatenate((A.T, zeros), axis=0) #(M + N) * M 
    left_matrix = np.concatenate((left_matrix_left, left_matrix_right), axis=1) #(M + N) * (M + N)
    zeros = np.zeros((M), dtype=np.float64)
    right_matrix = np.concatenate((-dfx, zeros), axis=0) #(M + N)
    direction = np.matmul(np.linalg.inv(left_matrix), right_matrix)
    newton_direction = direction[0:N]
    return newton_direction

def get_u_best(P, q, A, b, best_x):
    """Get the best u, which is the best result of the dual problem

    Args:
        P [numpy double array], [N * N]: [P]
        q [numpy double array], [N]: [q]
        A [numpy double array], [M * N]: [A]
        b [numpy double array], [M]: [b]
        best x [numpy double array], [N]: [the best x of the original problem]
  
    Returns:
        best_u [numpy double array], [M]: [best u of the dual problem]
    """
    a1 = -np.linalg.inv(np.matmul(A, A.T))
    a2 = np.matmul(A, np.matmul(P, best_x) + q)
    best_u = np.matmul(a1, a2)
    return best_u

def newton_method(P, q, A, b, x0, alpha, beta, epsilon):
    """Use newton method to optimize

    Args:
        P [numpy double array], [N * N]: [P]
        q [numpy double array], [N]: [q]
        A [numpy double array], [M * N]: [A]
        b [numpy double array], [M]: [b]
        x0 [numpy double array], [N]: [starting point]
        alpha [double]: [the parameter in backtracking]
        beta [double]: [the parameter in backtracking]
        epsilon [double]: [the stopping criterion]
    
    Returns:
        x_best [numpy double array], [N]: [the best x]
        u_best [numpy double array], [M]: [the best u]
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
    P, q, A, b = load_data()
    x0 = get_x0(A, b)
    alpha = 0.4
    beta = 0.5
    epsilon = 1e-5 
    x_best, u_best, y_best, log_dist_list, t_list = newton_method(P, q, A, b, x0, alpha, beta, epsilon)
    visualize(log_dist_list, t_list)
    save_data(x_best, u_best)
    print('y best:', y_best)

if __name__ == "__main__":
    main()