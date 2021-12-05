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
        self.n = 16
        self.alpha = 1000
        self.stop_criterion = 2e-2
        self.load_data()
        x_list, fx_list = self.proximal_iteration()
        best_x = x_list[-2]
        best_y = fx_list[-2]
        self.save_data(best_x, best_y)
        self.visualize(x_list, fx_list)


    def load_data(self):
        """Load the data
        """
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        a_place = os.path.join(data_dir, '1A.csv')
        b_place = os.path.join(data_dir, '1b.csv')
        self.A = np.zeros((self.n, self.n), dtype=np.float64)
        self.b = np.zeros((self.n, 1), dtype=np.float64)
        

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

    def save_data(self, best_x, best_y):
        """save the data

        Args:
            best_x [numpy double array], [N * 1]: [the best x]
            best_y [double]: [the best y]
        """
        current_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result')
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
        current_dir = os.path.join(current_dir, 'Q2')
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
    
        x_place = os.path.join(current_dir, 'x.csv')
        with open(x_place, 'w', encoding="utf-8", newline="") as csvfile:
            data_writer = csv.writer(csvfile)
            data_writer.writerows(best_x)
        print('The best y is', best_y)

    def f(self, x):
        """Get fx

        Args:
            x [numpy double array], [N * 1]: [x]
            
        Returns:
            fx [double]: [f(x)]
        """
        fx = np.linalg.norm(np.matmul(self.A, x) - self.b, ord=2)
        fx = fx ** 2 / 2 
        return fx
                    
    def get_new_x(self, x):
        """Get the new x

        Args:
            x [numpy double array], [N * 1]: [x]
            
        Returns:
            new_x [numpy double array], [N * 1]: [new x]
        """
        new_front = np.linalg.inv(np.matmul(self.A.T, self.A) * self.alpha + np.identity(self.n, dtype=np.float64))
        new_end = np.matmul(self.A.T, self.b) * self.alpha + x 
        new_x = np.matmul(new_front, new_end)
        return new_x
    
    def proximal_iteration(self):
        """The proximal iteration algorithm
        
        Returns:
            x_list [array of numpy float array]: [the list of x]
            fx_list [array of numpy float array]: [the list of fx]
        """ 
        x0 = np.zeros((self.n, 1), dtype=np.float64)
        x = x0 
        x_list = []
        fx_list = []
        additional_steps = 0
        while True:
            fx = self.f(x)
            x_list.append(x)
            fx_list.append(fx)
            
            if fx <= self.stop_criterion:
                if additional_steps >= 1:
                    break 
                additional_steps += 1

            x = self.get_new_x(x)
        return x_list, fx_list
    
    def visualize(self, x_list, fx_list):
        """Visualize the result

        Args:
            x_list [array of numpy float array]: [the list of x]
            fx_list [array of numpy float array]: [the list of fx]
        """
        best_x = x_list[-1] #N * 1
        k = len(x_list) - 1
        k_list = np.arange(1, k + 1, dtype=np.float64) #k
        x = np.array(x_list, dtype=np.float64)[:-1, :, :] #K * N * 1
        fx = np.array(fx_list, dtype=np.float64)[:-1] #K
        best_x = best_x.reshape(1, self.n, 1).repeat(k, axis=0) #K * N * 1
        dx = x - best_x #K * N * 1
        dx = dx.reshape(k, self.n) #K * N
        dx = np.linalg.norm(dx, axis=1, ord=2, keepdims=False) #K
        log_fx = np.log(fx)
        log_dx = np.log(dx)
        log_k = np.log(k_list)
        
        plt.xlabel('log(k)')
        plt.ylabel('log(||x_k - x*||_2)')
        plt.plot(log_k, log_dx)
        plt.show()

        plt.xlabel('log(k)')
        plt.ylabel('log(f(x_k))')
        plt.plot(log_k, log_fx)
        plt.show()
        
        
if __name__ == "__main__":
    a = Solver()
                    
            
        