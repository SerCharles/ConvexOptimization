import os
import csv
from scipy.io import loadmat

def load_data():
    """Load the data

    Returns:
        A [numpy double array], [m * n]: [A]
        b [numpy double array], [m * 1]: [b]
        lambda_0 [numpy double array], [n * 1]: [the initial lambda]
        mu_0 [numpy double array], [m * 1]: [the initial mu]
        P [numpy double array], [n * n]: [P]
        q [numpy double array], [n * 1]: [q]
        x_0 [numpy double array], [n * 1]: [the initial x]
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    A = loadmat(os.path.join(data_dir, 'A.mat'))['A']
    b = loadmat(os.path.join(data_dir, 'b.mat'))['b']
    lambda_0 = loadmat(os.path.join(data_dir, 'lambda.mat'))['lambda']
    mu_0 = loadmat(os.path.join(data_dir, 'mu.mat'))['mu']
    P = loadmat(os.path.join(data_dir, 'P.mat'))['P']
    q = loadmat(os.path.join(data_dir, 'q.mat'))['q']
    x0 = loadmat(os.path.join(data_dir, 'x_0.mat'))['x_0']
    return A, b, lambda_0, mu_0, P, q, x0

def save_data(method, x, lambda_, mu, y):
    """Save the results

    Args:
        method [string]: [the method name]
        x [numpy double array], [n * 1]: [the best x]
        lambda_ [numpy double array], [n * 1]: [the best lambda]
        mu [numpy double array], [m * 1]: [the best mu]
        y [double]: [the best y]
    """
    print('The best y is', y)
    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    result_dir = os.path.join(result_dir, method)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    with open(os.path.join(result_dir, 'x.csv'), 'w', encoding="utf-8", newline="") as csvfile:
        data_writer = csv.writer(csvfile)
        data_writer.writerows(x)
    with open(os.path.join(result_dir, 'lambda.csv'), 'w', encoding="utf-8", newline="") as csvfile:
        data_writer = csv.writer(csvfile)
        data_writer.writerows(lambda_)
    with open(os.path.join(result_dir, 'mu.csv'), 'w', encoding="utf-8", newline="") as csvfile:
        data_writer = csv.writer(csvfile)
        data_writer.writerows(mu)