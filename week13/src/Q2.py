import os
import csv
from math import *
import numpy as np
import matplotlib.pyplot as plt


class Solver(object):
    """The solver of Q1
    """
    def __init__(self):
        """init and run
        """
        self.m = 60
        self.n = 100
        self.stop_criterion = 1e-5
        self.x0 = np.zeros((self.n, 1), dtype=np.float64)
        self.load_data()
        self.get_x0()
        self.get_P()

    def load_data(self):
        """Load the data
        """
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'problem2data')
        a_place = os.path.join(data_dir, 'A.csv')
        b_place = os.path.join(data_dir, 'b.csv')
        self.A = np.zeros((self.m, self.n), dtype=np.float64)
        self.b = np.zeros((self.m, 1), dtype=np.float64)

        with open(a_place, 'r', encoding="utf-8-sig") as csvfile:
            data_reader = csv.reader(csvfile)
            for row in data_reader:
                i = data_reader.line_num - 1 
                for j in range(len(row)):
                    self.A[i, j] = float(row[j])
    
        with open(b_place, 'r', encoding="utf-8-sig") as csvfile:
            data_reader = csv.reader(csvfile)
            for row in data_reader:
                i = data_reader.line_num - 1 
                for j in range(len(row)):
                    self.b[i, j] = float(row[j])
                    

    def save_data(self, id, best_x, best_y):
        """save the data

        Args:
            id [int]: [the type of the method]
            best_x [numpy double array], [N * 1]: [the best x]
            best_y [double]: [the best y]
        """
        current_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result')
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
        current_dir = os.path.join(current_dir, 'Q2')
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
    
        x_place = os.path.join(current_dir, 'x' + str(id) + '.csv')
        with open(x_place, 'w', encoding="utf-8", newline="") as csvfile:
            data_writer = csv.writer(csvfile)
            data_writer.writerows(best_x)
        print('The best y is', best_y)
    

    def get_x0(self):
        """Get the x0 that satisfies Ax0 = b
        """
        A_down_left = np.zeros((self.n - self.m, self.m), dtype=np.float64) 
        A_down_right = np.identity(self.n - self.m, dtype=np.float64)
        A_down = np.concatenate((A_down_left, A_down_right), axis=1) #(n - m) * n
        A_extend = np.concatenate((self.A, A_down), axis=0) #n * n
        b_down = np.zeros((self.n - self.m, 1), dtype=np.float64)
        b_extend = np.concatenate((self.b, b_down), axis=0) #n * 1
        x0 = np.matmul(np.linalg.inv(A_extend), b_extend) #n * 1
        self.x0 = x0
    
    def get_P(self):
        """Get the P used in the third method
        """
        hessian = np.matmul(self.A.T, self.A)
        u, sigma, v = np.linalg.svd(hessian)
        self.M = max(abs(sigma[0]), abs(sigma[-1]))
        sigma_diagonal = np.diag(sigma)
        M_diagonal = np.identity(self.n) * (self.M + 1)
        self.P = np.matmul(np.sqrt(M_diagonal - sigma_diagonal), v)
        
    
    def solver_1(self):
        """The first solving method
        
        Returns:
            x_list [array of numpy float array], [N * 1 each]: [the list of x]
        """
        x0 = self.x0
        y0 = x0
        z0 = x0
        alpha = 1e-3
        addition_steps = 0
        x_list = [] 
        x = x0
        y = y0 
        z = z0 
        x_list.append(x0)
        while True:
            last_x = x 
            last_y = y 
            last_z = z 
            ataat = np.matmul(self.A.T, np.linalg.inv(np.matmul(self.A, self.A.T)))
            x = last_z - np.matmul(ataat, (np.matmul(self.A, last_z) - self.b))
            v = x * 2 - last_z 
            y = (v - alpha) * (v > alpha) + (v + alpha) * (v < -alpha)
            z = last_z + y - x 
            x_list.append(x)
            dist = np.linalg.norm(last_x, ord=1) - np.linalg.norm(x, ord=1)
            if dist < self.stop_criterion:
                if addition_steps >= 1:
                    break 
                else:
                    addition_steps += 1
        return x_list
    
    def solver_2(self):
        """The second solving method
        
        Returns:
            x_list [array of numpy float array], [N * 1 each]: [the list of x]
        """
        x0 = self.x0
        y0 = x0
        u0 = np.zeros((self.n, 1), dtype=np.float64)
        v0 = np.zeros((self.m, 1), dtype=np.float64)
        alpha = 1000
        addition_steps = 0
        x_list = [] 
        x = x0
        y = y0 
        u = u0 
        v = v0
        x_list.append(x0)
        while True:
            last_x = x 
            last_y = y 
            last_u = u 
            last_v = v 
            s = last_y - last_u / alpha 
            x = (s - 1 / alpha) * (s > (1 / alpha)) + (s + 1 / alpha) * (s < (-1 / alpha))
            y1 = np.linalg.inv(np.matmul(self.A.T, self.A) + np.identity(self.n))
            y2 = np.matmul(self.A.T, self.b) + x + last_u / alpha - np.matmul(self.A.T, last_v) / alpha
            y = np.matmul(y1, y2)
            u = last_u + (x - y) * alpha
            v = last_v + (np.matmul(self.A, y) - self.b) * alpha
            x_list.append(x)
            dist = np.linalg.norm(last_x, ord=1) - np.linalg.norm(x, ord=1)
            if dist < self.stop_criterion:
                if addition_steps >= 1:
                    break 
                else:
                    addition_steps += 1
        return x_list
    
    def solver_3(self):
        """The second solving method
        
        Returns:
            x_list [array of numpy float array], [N * 1 each]: [the list of x]
        """
        x0 = self.x0
        y0 = np.matmul(self.P, self.x0)
        u0 = np.zeros((self.m, 1), dtype=np.float64)
        v0 = np.zeros((self.n, 1), dtype=np.float64)
        alpha = 10000 / (self.M + 1)
        addition_steps = 0
        x_list = [] 
        x = x0
        y = y0 
        u = u0 
        v = v0
        x_list.append(x0)
        while True:
            last_x = x 
            last_y = y 
            last_u = u 
            last_v = v 
            s = np.matmul(self.A.T, self.b) * alpha + np.matmul(self.P.T, last_y) * alpha + np.matmul(self.P.T, last_v) - np.matmul(self.A.T, last_u)
            x = (s - 1) / (alpha * (self.M + 1)) * (s > 1) + (s + 1) / (alpha * (self.M + 1)) * (s < -1)
            y = np.matmul(self.P, x) - last_v / alpha
            u = last_u + (np.matmul(self.A, x) - self.b) * alpha
            v = last_v + (y - np.matmul(self.P, x)) * alpha
            x_list.append(x)
            dist = np.linalg.norm(last_x, ord=1) - np.linalg.norm(x, ord=1)
            if dist < self.stop_criterion:
                if addition_steps >= 1:
                    break 
                else:
                    addition_steps += 1
        return x_list
    
    def visualize(self, x_list):
        """Visualize the results

        Args:
            x_list [array of numpy float array], [N * 1 each]: [the list of x]
        """
        best_x = x_list[-1].reshape(1, self.n) #1 * N
        x = np.concatenate(x_list[1:-1], axis=1).T #K * N
        k = x.shape[0]
        k_list = np.arange(1, k + 1)
        dist_x = np.linalg.norm(x - best_x.repeat(k, axis=0), ord=2, axis=1) #K
        size_x = np.linalg.norm(x, ord=1, axis=1) #K 
        log_k = np.log(k_list)
        log_size_x = np.log(size_x)
        log_dist_x = np.log(dist_x)
        
        plt.xlabel('log(k)')
        plt.ylabel('log(||x_k - x*||_2)')
        plt.plot(log_k, log_dist_x)
        plt.show()
        
        plt.xlabel('log(k)')
        plt.ylabel('log(||x_k||_1)')
        plt.plot(log_k, log_size_x)
        plt.show()
        
    def run(self):
        """The main function
        """
        
        x_list = self.solver_1()
        best_x = x_list[-1]
        print('normal =', np.linalg.norm(np.matmul(self.A, best_x) - self.b, ord=2))
        best_y = np.linalg.norm(best_x, ord=1)
        self.save_data(1, best_x, best_y)
        self.visualize(x_list)
        
        x_list = self.solver_2()
        best_x = x_list[-1]
        print('normal =', np.linalg.norm(np.matmul(self.A, best_x) - self.b, ord=2))
        best_y = np.linalg.norm(best_x, ord=1)
        self.save_data(2, best_x, best_y)
        self.visualize(x_list)
        
        x_list = self.solver_3()
        best_x = x_list[-1]
        print('normal =', np.linalg.norm(np.matmul(self.A, best_x) - self.b, ord=2))
        best_y = np.linalg.norm(best_x, ord=1)
        self.save_data(3, best_x, best_y)
        self.visualize(x_list)
        
if __name__ == "__main__":
    a = Solver()
    a.run()