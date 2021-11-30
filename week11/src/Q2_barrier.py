from math import *
import numpy as np
import matplotlib.pyplot as plt
from data import *

class BarrierMethod(object):
    """Barrier method
    """
    
    def __init__(self):
        """Init the parameters and data
        """
        self.u = 10.0 
        self.stop_criterion = 1e-8
        self.A, self.b, lambda0, mu0, self.P, self.q, x0 = load_data()
        self.m = self.A.shape[0]
        self.n = self.A.shape[1]


    def get_x0(self):
        """Get the x0 that satisfies Ax0 = b

        Returns:
            x0 [numpy double array], [n * 1]: [starting point]
        """

        A_down_left = np.zeros((self.n - self.m, self.m), dtype=np.float64) 
        A_down_right = np.identity(self.n - self.m, dtype=np.float64)
        A_down = np.concatenate((A_down_left, A_down_right), axis=1) #(n - m) * n
        A_extend = np.concatenate((self.A, A_down), axis=0) #n * n
        b_down = np.zeros((self.n - self.m, 1), dtype=np.float64)
        b_extend = np.concatenate((self.b, b_down), axis=0) #n * 1
        x0 = np.matmul(np.linalg.inv(A_extend), b_extend) #n * 1
        return x0

    def g(self, x, t):
        """Get the value of tf(x) + phi(x)

        Args:
            x [numpy double array], [n * 1]: [x]
            t [double]: [t]
    
        Returns:
            gx [double]: [tf(x) + phi(x)]
        """
        gx = np.matmul(x.T, np.matmul(self.P, x)) / 2 * t + np.matmul(self.q.T, x) * t
        log_x = np.log(x)
        gx = gx - np.sum(log_x)
        return gx 

    def dg(self, x, t):
        """Get the gradient of tf(x) + phi(x)

        Args:
            x [numpy double array], [n * 1]: [x]
            t [double]: [t]
    
        Returns:
            dgx [numpy double array], [n * 1]: [the gradient of tf(x) + phi(x)]
        """
        dgx = (np.matmul(self.P, x) + self.q) * t - 1.0 / x 
        return dgx 

    def d2g(self, x, t):
        """Get the Hessian of tf(x) + phi(x)

        Args:
            x [numpy double array], [n * 1]: [x]
            t [double]: [t]
    
        Returns:
            d2gx [numpy double array], [n * n]: [the Hessian of tf(x) + phi(x)]
        """
        d2gx = self.P * t 
        for i in range(self.n):
            d2gx[i, i] = d2gx[i, i] + (1 / x[i, 0] / x[i, 0])
        return d2gx

    def g0(self, xs, t):
        """Get the value of ts + phi(x), used in the first step

        Args:
            xs [numpy double array], [(n + 1) * 1]: [x and s]
            t [double]: [t]
    
        Returns:
            g0x [double]: [tf(x) + phi(x)]
        """
        s = xs[self.n, 0]
        x = xs[0:self.n, :]
        g0x = s * t - np.sum(np.log(x + s))
        return g0x

    def dg0(self, xs, t):
        """Get the gradient of ts + phi(x), used in the first step

        Args:
            xs [numpy double array], [(n + 1) * 1]: [x and s]
            t [double]: [t]
    
        Returns:
            dg0x [numpy double array], [(n + 1) * 1]: [the gradient of tf(x) + phi(x)]
        """
        s = xs[self.n, 0]
        x = xs[0:self.n, :]
        dg0x_front = -1.0 / (x + s) #n * 1
        dg0x_back = t + np.sum(dg0x_front)
        dg0x_back = np.array([dg0x_back]).reshape(1, 1)
        dg0x = np.concatenate((dg0x_front, dg0x_back), axis=0) #(n + 1) * 1
        return dg0x 

    def d2g0(self, xs, t):
        """Get the gradient of ts + phi(x), used in the first step

        Args:
            xs [numpy double array], [(n + 1) * 1]: [x and s]
            t [double]: [t]
    
        Returns:
            d2g0x [numpy double array], [(n + 1) * (n + 1)]: [the gradient of tf(x) + phi(x)]
        """
        s = xs[self.n, 0]
        x = xs[0:self.n, :]
        d2g0x_main = np.zeros((self.n, self.n), dtype=np.float64)
        for i in range(self.n):
            d2g0x_main[i, i] = 1 / ((s + x[i, 0]) ** 2)
        m = 1.0 / ((x + s) ** 2) #n * 1
        d2g0x_last = np.array([np.sum(m)]).reshape(1, 1)
        d2g0x_up = np.concatenate((d2g0x_main, m), axis=1) #n * (n + 1)
        d2g0x_down = np.concatenate((m.T, d2g0x_last), axis=1) #1 * (n + 1)
        d2g0x = np.concatenate((d2g0x_up, d2g0x_down), axis=0) #(n + 1) * (n + 1)
        return d2g0x

    def newton_first_step(self, x0, t, alpha=0.4, beta=0.5, stop_criterion=1e-12):
        """
        Args:
            x0 [numpy double array], [n * 1]: [the initial x0]
            t [double]: [the current t]
            alpha [double]: [the parameter in backtracking]
            beta [double]: [the parameter in backtracking]
            stop_criterion [double]: [the stop criterion]
    
        Returns:
            result_x [numpy double array], [n * 1]: [the initial x0 used in the final newton method]
        """

        s = -np.min(x0) + 1
        xs0 = np.concatenate((x0, np.array([s]).reshape(1, 1)), axis=0) #(n + 1) * 1
        xs = xs0
        expanded_A = np.concatenate((self.A, np.zeros((self.m, 1), dtype=np.float64)), axis=1) #m * (n + 1)
        while True:
            g0xs = self.g0(xs, t)
            dg0xs = self.dg0(xs, t)
            d2g0xs = self.d2g0(xs, t)
        
            KKT_front_up = np.concatenate((d2g0xs, expanded_A.T), axis=1) #(n + 1) * (n + m + 1)
            KKT_front_down = np.concatenate((expanded_A, np.zeros((self.m, self.m), dtype=np.float64)), axis=1) #m * (n + m + 1)
            KKT_front = np.concatenate((KKT_front_up, KKT_front_down), axis=0) #(n + m + 1) * (n + m + 1)
            KKT_end = np.concatenate((-dg0xs, np.zeros((self.m, 1), dtype=np.float64)), axis=0) #(n + m + 1) * 1
            newton_direction = np.matmul(np.linalg.inv(KKT_front), KKT_end)[0:self.n + 1, :] #(n + 1) * 1
            lambda_2 = float(-np.matmul(dg0xs.T, newton_direction))
        
            if lambda_2 < stop_criterion:
                result_x = xs[0:self.n, :]
                s = xs[self.n, 0]
                break 
        
            learning_rate = 1.0
            while True:
                dist = self.g0(xs + learning_rate * newton_direction, t) - g0xs - alpha * learning_rate * float(np.matmul(dg0xs.T, newton_direction))
                if dist <= 0:
                    break 
                learning_rate = beta * learning_rate
            xs = xs + learning_rate * newton_direction
    
        return result_x       

    def first_step(self, x0):
        """The first step method, used in getting the start point

        Args:
            x0 [numpy double array], [n * 1]: [the initial x0]
            
        Returns:
            x [numpy double array], [n * 1]: [the initial x0 used in the final newton method]
        """
        t = 1.0
        x = x0
        while True:
            if self.n / t < self.stop_criterion:
                break
            x = self.newton_first_step(x, t)
            t = self.u * t
        return x


    def newton_method(self, x0, t, alpha=0.4, beta=0.5, stop_criterion=1e-12):
        """
        Args:
            x0 [numpy double array], [n * 1]: [the initial x0]
            t [double]: [the current t]
            alpha [double]: [the parameter in backtracking]
            beta [double]: [the parameter in backtracking]
            stop_criterion [double]: [the stop criterion]
    
        Returns:
            best_x [numpy double array], [n * 1]: [the best x of the iteration]
            best_lambda [numpy double array], [n * 1]: [the best lambda of the iteration]
            best_mu [numpy double array], [m * 1]: [the best mu of the iteration]
            best_y [double]: [the best y of the iteration]
            iter_times [int]: [the iteration times]
        """

        x = x0
        t = 1.0
        iter_times = 0
        while True:
            gx = self.g(x, t)
            dgx = self.dg(x, t)
            d2gx = self.d2g(x, t)
            KKT_front_up = np.concatenate((d2gx, self.A.T), axis=1) #n * (n + m)
            KKT_front_down = np.concatenate((self.A, np.zeros((self.m, self.m), dtype=np.float64)), axis=1) #m * (n + m)
            KKT_front = np.concatenate((KKT_front_up, KKT_front_down), axis=0) #(m + n) * (m + n)
            KKT_end = np.concatenate((-dgx, np.zeros((self.m, 1), dtype=np.float64)), axis=0) #(m + n) * 1
            KKT_result = np.matmul(np.linalg.inv(KKT_front), KKT_end) #(m + n) * 1
            newton_direction = KKT_result[0:self.n, :] #n * 1
            best_mu = KKT_result[self.n:, :] #m * 1
            lambda_2 = float(-np.matmul(dgx.T, newton_direction))
        
            if lambda_2 < stop_criterion:
                best_x = x
                break 
        
            learning_rate = 1.0
            while True:
                dist = self.g(x + learning_rate * newton_direction, t) - gx - alpha * learning_rate * float(np.matmul(dgx.T, newton_direction))
                if dist <= 0:
                    break 
                learning_rate = beta * learning_rate
            x = x + learning_rate * newton_direction
            iter_times += 1
    
        best_lambda = np.matmul(self.P, best_x) + self.q.T - np.matmul(self.A.T, best_mu)
        best_y = self.g(x, t)
        return best_x, best_lambda, best_mu, best_y, iter_times

    def barrier_method(self, x0):
        """The barrier method

        Args:
            x0 [numpy double array], [n * 1]: [the initial x0]

        Returns:
            best_x [numpy double array], [n * 1]: [the best x]
            best_lambda [numpy double array], [n * 1]: [the best lambda]
            best_mu [numpy double array], [m * 1]: [the best mu]    
            best_y [double]: [the best y]
            log_gap_list [float array]: [the list of log(n / t)]      
            iter_time_list [int array]: [the list of the newton iteration times] 
        """
        t = 1.0
        x = x0
        log_gap_list = []
        iter_time_list = []
        while True:
            if self.n / t < self.stop_criterion:
                best_x = x 
                best_lambda = lambda_ 
                best_mu = mu
                best_y = y
                break
            x, lambda_, mu, y, iter_times = self.newton_method(x, t)
            t = self.u * t
            log_gap_list.append(log(self.n / t))
            iter_time_list.append(iter_times)
        return best_x, best_lambda, best_mu, best_y, log_gap_list, iter_time_list

    def visualize(self, log_gap_list, iter_time_list):
        """Visualize the relationship between log(n / t) and the newton iteration times k

        Args:
            log_gap_list [float array]: [the list of log(n / t)]      
            iter_time_list [int array]: [the list of the newton iteration times] 
        """
        log_gap = np.array(log_gap_list)
        iter_time = np.array(iter_time_list)
        plt.xlabel('log(n / t)')
        plt.ylabel('iter_time')
        plt.scatter(log_gap, iter_time, color='b', zorder=1)
        plt.plot(log_gap, iter_time, color='b', zorder=2)
        plt.show()
        
    def main(self):
        """The main process of barrier method
        """     
        x0 = self.get_x0()
        start_x = self.first_step(x0)
        best_x, best_lambda, best_mu, best_y, log_gap_list, iter_time_list = self.barrier_method(start_x)
        save_data('barrier', best_x, best_lambda, best_mu, float(best_y))
        self.visualize(log_gap_list, iter_time_list)




if __name__ == "__main__":
    a = BarrierMethod()
    a.main()