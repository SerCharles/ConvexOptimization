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
        self.n = 100
        self.stop_criterion = 1e-5
        self.x0 = np.zeros((self.n, 1), dtype=np.float64)
        self.load_data()
        self.get_m()


    def load_data(self):
        """Load the data
        """
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'problem1data')
        a1_place = os.path.join(data_dir, 'A1.csv')
        b1_place = os.path.join(data_dir, 'b1.csv')
        a2_place = os.path.join(data_dir, 'A2.csv')
        b2_place = os.path.join(data_dir, 'b2.csv')
        self.A1 = np.zeros((self.n, self.n), dtype=np.float64)
        self.b1 = np.zeros((self.n, 1), dtype=np.float64)
        self.A2 = np.zeros((self.n, self.n), dtype=np.float64)
        self.b2 = np.zeros((self.n, 1), dtype=np.float64)

        with open(a1_place, 'r', encoding="utf-8-sig") as csvfile:
            data_reader = csv.reader(csvfile)
            for row in data_reader:
                i = data_reader.line_num - 1 
                for j in range(len(row)):
                    self.A1[i, j] = float(row[j])
    
        with open(b1_place, 'r', encoding="utf-8-sig") as csvfile:
            data_reader = csv.reader(csvfile)
            for row in data_reader:
                i = data_reader.line_num - 1 
                for j in range(len(row)):
                    self.b1[i, j] = float(row[j])
                    
        with open(a2_place, 'r', encoding="utf-8-sig") as csvfile:
            data_reader = csv.reader(csvfile)
            for row in data_reader:
                i = data_reader.line_num - 1 
                for j in range(len(row)):
                    self.A2[i, j] = float(row[j])
    
        with open(b2_place, 'r', encoding="utf-8-sig") as csvfile:
            data_reader = csv.reader(csvfile)
            for row in data_reader:
                i = data_reader.line_num - 1 
                for j in range(len(row)):
                    self.b2[i, j] = float(row[j])

    def save_data(self, best_x1, best_y1, best_x2, best_y2):
        """save the data

        Args:
            best_x1 [numpy double array], [N * 1]: [the best x of A1, b1]
            best_y1 [double]: [the best y of A1, b1]
            best_x2 [numpy double array], [N * 1]: [the best x of A2, b2]
            best_y2 [double]: [the best y of A2, b2]
        """
        current_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result')
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
        current_dir = os.path.join(current_dir, 'Q2')
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
    
        x1_place = os.path.join(current_dir, 'x1.csv')
        x2_place = os.path.join(current_dir, 'x2.csv')
        with open(x1_place, 'w', encoding="utf-8", newline="") as csvfile:
            data_writer = csv.writer(csvfile)
            data_writer.writerows(best_x1)

        with open(x2_place, 'w', encoding="utf-8", newline="") as csvfile:
            data_writer = csv.writer(csvfile)
            data_writer.writerows(best_x2)
        
        print('The best y1 is', best_y1)
        print('The best y2 is', best_y2)
    
    def get_m(self):
        """Get the M1 and M2 corresponding to A1, A2
        """
        hessian_1 = np.matmul(self.A1.T, self.A1)
        u, sigma, v = np.linalg.svd(hessian_1)
        self.M1 = max(abs(sigma[0]), abs(sigma[-1]))
        self.alpha_1 = 1 / self.M1
        print('M1 =', self.M1)
        hessian_2 = np.matmul(self.A2.T, self.A2)
        u, sigma, v = np.linalg.svd(hessian_2)
        self.M2 = max(abs(sigma[0]), abs(sigma[-1]))
        self.alpha_2 = 1 / self.M2
        print('M2 =', self.M2)
    
    def h(self, x):
        """Get h(x) = f(x) + g(x)

        Args:
            x [numpy double array], [N * 1]: [x]
            
        Returns:
            hx [double]: [h(x)]
        """
        hx1 = np.linalg.norm(np.matmul(self.A, x) - self.b, ord=2) ** 2 / 2
        hx2 = np.linalg.norm(x, ord=1)
        hx = hx1 + hx2
        return hx 
    
    def next_x(self, x):
        """Get the next x

        Args:
            x [numpy double array], [N * 1]: [x]
            
        Returns:
            next_x [numpy double array], [N * 1]: [next x]
        """
        v = x - np.matmul(np.matmul(self.A.T, self.A), x) * self.alpha + \
            np.matmul(self.A.T, self.b) * self.alpha
        next_x = (v - self.alpha) * (v > self.alpha) + (v + self.alpha) * (v < -self.alpha)
        return next_x
    
    def ista_algorithm(self):
        """The main algorithm of ista algorithm
        
        Returns:
            x_list [array of numpy double array], [1 * N each]:[the list of x]
            hx_list [array of double]: [the list of h(x)]
        """
        x_list = []
        hx_list = []
        x = self.x0 
        iter_time = 0
        additional_steps = 0
        while True:
            hx = self.h(x)
            x_list.append(x.reshape(1, self.n))
            hx_list.append(hx)
            if iter_time != 0:
                dist = abs(hx - last_hx)
                if dist < self.stop_criterion:
                    if additional_steps >= 1:
                        break 
                    else:
                        additional_steps += 1
            last_x = x 
            last_hx = hx 
            iter_time += 1
            x = self.next_x(x)
        return x_list, hx_list
    
    def visualize(self, x_list, hx_list):
        """Visualize the results

        Args:
            x_list [array of numpy double array], [1 * N each]:[the list of x]
            hx_list [array of double]: [the list of h(x)]
        """
        k = len(x_list) - 2
        k_list = np.arange(1, len(x_list) - 1)
        x = np.concatenate(x_list[1:-1], axis=0)
        hx = np.array(hx_list[1:-1])
        best_x = x_list[-1].reshape(1, self.n).repeat(k, axis=0)
        hx_last = np.array(hx_list[0:-2])
        dx = np.linalg.norm(x - best_x, ord=2, axis=1)
        dhx = hx_last - hx 
        log_k = np.log(k_list)
        log_dx = np.log(dx)
        log_hx = np.log(hx)
        log_dhx = np.log(dhx)
        plt.xlabel('log(k)')
        plt.ylabel('log(||x_k - x*||_2)')
        plt.plot(log_k, log_dx)
        plt.show()
        
        plt.xlabel('log(k)')
        plt.ylabel('log(h(x_k))')
        plt.plot(log_k, log_hx)
        plt.show()

        plt.xlabel('log(k)')
        plt.ylabel('log(h(x_k-1) - h(x_k))')
        plt.plot(log_k, log_dhx)
        plt.show()
    
    def run(self):
        """The main algorithm of solving
        """
        self.A = self.A1 
        self.b = self.b1 
        self.M = self.M1
        self.alpha = self.alpha_1
        x_list, hx_list = self.ista_algorithm()
        best_x1 = x_list[-1].reshape(self.n, 1)
        best_y1 = hx_list[-1]
        self.visualize(x_list, hx_list)
        self.A = self.A2 
        self.b = self.b2 
        self.M = self.M2
        self.alpha = self.alpha_2
        x_list, hx_list = self.ista_algorithm()
        best_x2 = x_list[-1].reshape(self.n, 1)
        best_y2 = hx_list[-1]

        self.visualize(x_list, hx_list)
        self.save_data(best_x1, best_y1, best_x2, best_y2)
        
if __name__ == "__main__":
    a = Solver()
    a.run()