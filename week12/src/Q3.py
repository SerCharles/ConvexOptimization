import os
import csv
from math import *
import numpy as np
import matplotlib.pyplot as plt


class Solver(object):
    """The solver of Q2
    """
    def __init__(self):
        """init and run
        """
        self.n = 10
        self.c = 0.01
        self.beta = 0.5
        self.stop_criterion = 1e-8
        self.load_data()
        self.get_m()
        x_list, dx_list, hx_list = self.subgradient_descent(1)
        best_x1 = x_list[-2]
        best_y1 = hx_list[-2]
        self.visualize(x_list, dx_list, hx_list)
        x_list, dx_list, hx_list = self.subgradient_descent(2)
        best_x2 = x_list[-2]
        best_y2 = hx_list[-2]
        self.visualize(x_list, dx_list, hx_list)
        self.save_data(best_x1, best_y1, best_x2, best_y2)

    def load_data(self):
        """Load the data
        """
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        a_place = os.path.join(data_dir, '2A.csv')
        b_place = os.path.join(data_dir, '2b.csv')
        x0_place = os.path.join(data_dir, '2x0.csv')
        self.A = np.zeros((self.n, self.n), dtype=np.float64)
        self.b = np.zeros((self.n, 1), dtype=np.float64)
        self.x0 = np.zeros((self.n, 1), dtype=np.float64)

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
                    
        with open(x0_place, 'r', encoding="utf-8-sig") as csvfile:
            data_reader = csv.reader(csvfile)
            for row in data_reader:
                i = data_reader.line_num - 1 
                for j in range(len(row)):
                    self.x0[j, i] = float(row[j])

    def save_data(self, best_x1, best_y1, best_x2, best_y2):
        """save the data

        Args:
            best_x1 [numpy double array], [N * 1]: [the best x of method 1]
            best_y1 [double]: [the best y of method 1]
            best_x2 [numpy double array], [N * 1]: [the best x of method 2]
            best_y2 [double]: [the best y of method 2]
        """
        current_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result')
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
        current_dir = os.path.join(current_dir, 'Q3')
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
    
        x1_place = os.path.join(current_dir, 'x1.csv')
        with open(x1_place, 'w', encoding="utf-8", newline="") as csvfile:
            data_writer = csv.writer(csvfile)
            data_writer.writerows(best_x1)
        print('The best y of problem 1 is', best_y1)
        
        x2_place = os.path.join(current_dir, 'x2.csv')
        with open(x2_place, 'w', encoding="utf-8", newline="") as csvfile:
            data_writer = csv.writer(csvfile)
            data_writer.writerows(best_x2)
        print('The best y of problem 2 is', best_y2)
        
        
    def h(self, x):
        """Get h(x), f(x) + Lasso(x)

        Args:
            x [numpy double array], [N * 1]: [x]
            
        Returns:
            hx [double]: [h(x), f(x) + Lasso(x)]
        """
        fx = np.linalg.norm(np.matmul(self.A, x) - self.b, ord=2)
        fx = fx ** 2 / 2
        x1 = np.sum(np.abs(x))
        fx = fx + x1
        return fx 
    
    def dh(self, x):
        """Get dh(x), h(x) = f(x) + Lasso(x)

        Args:
            x [numpy double array], [N * 1]: [x]
            
        Returns:
            dhx [numpy double array], [N * 1]: [dh(x), h(x) = f(x) + Lasso(x)]
        """
        dhx = np.matmul(self.A.T, np.matmul(self.A, x)) - np.matmul(self.A.T, self.b) 
        sign_x = ((x > 0).astype(np.float64) - (x < 0).astype(np.float64))
        dhx = dhx + sign_x 
        return dhx
    
    def get_m(self):
        """Get m, which is the min eigenvalue of A^TA
        """
        hessian = np.matmul(self.A.T, self.A)
        u, sigma, v = np.linalg.svd(hessian)
        self.m = sigma[-1]
        print('m =', self.m)
        

    
    def get_learning_rate(self, k, type):
        """Get the learning rate 

        Args:
            k [int]: [the iteration times, starting from 1]
            type [int]: [type 1 or type 2]
            
        Returns:
            learning_rate [double]: [the learning rate]
        """
        if type == 1:
            learning_rate = self.c * (k ** (-self.beta))
        else:
            learning_rate = 1 / self.m / k
        return learning_rate    
    
    def subgradient_descent(self, type):
        """The subgradient descent method
        Args:
            type [int]: [type 1 or type 2]
        
        Returns:
            x_list [array of numpy double array]: [the list of x]
            dx_list [array of numpy double array]: [the list of x_k+1 - x_k]
            hx_list [array of numpy double array]: [the list of hx]
        """
        x = self.x0
        iter_time = 0
        x_list = []
        dx_list = []
        hx_list = []
        additional_steps = 0
        while True:
            hx = self.h(x)
            dhx = self.dh(x)
            x_list.append(x)
            hx_list.append(hx)
            if iter_time > 0:
                dx = np.linalg.norm(x - last_x, ord=2)
                dx_list.append(dx)
                if dx ** 2 < self.stop_criterion:
                    if additional_steps >= 1:
                        break 
                    else:
                        additional_steps += 1
                
            iter_time += 1                
            learning_rate = self.get_learning_rate(iter_time, type)
            last_x = x 
            x = x - dhx * learning_rate
        return x_list, dx_list, hx_list
    
    def visualize(self, x_list, dx_list, hx_list):
        """Visualize the result

        Args:
            x_list [array of numpy double array]: [the list of x]
            dx_list [array of numpy double array]: [the list of x_k+1 - x_k]
            hx_list [array of numpy double array]: [the list of hx]        
        """
        best_x = x_list[-1] #N * 1
        k = len(x_list) - 1
        k_list = np.arange(1, k + 1, dtype=np.float64) #k
        x = np.array(x_list, dtype=np.float64)[:-1, :, :] #K * N * 1
        dx = np.array(dx_list, dtype=np.float64) #K
        hx = np.array(hx_list, dtype=np.float64)[:-1] #K
        best_x = best_x.reshape(1, self.n, 1).repeat(k, axis=0) #K * N * 1
        dx_best = x - best_x #K * N * 1
        dx_best = dx_best.reshape(k, self.n) #K * N
        dx_best = np.linalg.norm(dx_best, axis=1, ord=2, keepdims=False) #K
        log_k = np.log(k_list)
        log_hx = np.log(hx)
        log_dx = np.log(dx)
        log_dx_best = np.log(dx_best)
        
        plt.xlabel('log(k)')
        plt.ylabel('log(||x_k+1 - x_k||_2)')
        plt.plot(log_k, log_dx)
        plt.show()
        
        plt.xlabel('log(k)')
        plt.ylabel('log(||x_k - x*||_2)')
        plt.plot(log_k, log_dx_best)
        plt.show()

        plt.xlabel('log(k)')
        plt.ylabel('log(h(x_k))')
        plt.plot(log_k, log_hx)
        plt.show()
        
if __name__ == "__main__":
    a = Solver()