from math import *
import numpy as np
import matplotlib.pyplot as plt
from data import *

class PrimalDualMethod(object):
    """Primal Dual method
    """
    def __init__(self):
        """Init the parameters and data
        """
        self.u = 10.0 
        self.stop_criterion = 1e-8
        self.A, self.b, self.lambda0, self.mu0, self.P, self.q, self.x0 = load_data()
        self.m = self.A.shape[0]
        self.n = self.A.shape[1]

    def f(self, x):
        """Get the value of f(x)

        Args:
            x [numpy double array], [n * 1]: [x]
    
        Returns:
            fx [double]: [f(x)]
        """
        fx = np.matmul(x.T, np.matmul(self.P, x)) / 2 + np.matmul(self.q.T, x)
        return fx 

    def df(self, x):
        """Get the value of df(x)

        Args:
            x [numpy double array], [n * 1]: [x]
        
        Returns:
            dfx [numpy double array], [n * 1]: [df(x)]
        """
        dfx = np.matmul(self.P, x) + self.q
        return dfx 
    
    def d2f(self, x):
        """Get the value of d2f(x)

        Args:
            x [numpy double array], [n * 1]: [x]
            
        Returns:
            d2fx [numpy double array], [n * n]: [d2f(x)]
        """
        return self.P
    
    def get_residual_dual(self, x, lambda_, mu, t):
        """Get the dual residual

        Args:
            x [numpy double array], [n * 1]: [x]
            lambda_ [numpy double array], [n * 1]: [lambda]
            mu [numpy double array], [m * 1]: [mu]
            t [double]: [parameter of constraints]

        Returns:
            r_dual [numpy double array], [n * 1]: [the dual residual]
        """
        r_dual = self.df(x) - lambda_ + np.matmul(self.A.T, mu)
        return r_dual
    
    def get_residual_center(self, x, lambda_, mu, t):
        """Get the center residual

        Args:
            x [numpy double array], [n * 1]: [x]
            lambda_ [numpy double array], [n * 1]: [lambda]
            mu [numpy double array], [m * 1]: [mu]
            t [double]: [parameter of constraints]
            
        Returns:
            r_cent [numpy double array], [n * 1]: [the center residual]
        """
        r_cent = x * lambda_ - (1.0 / t)
        return r_cent
    
    def get_residual_primal(self, x, lambda_, mu, t):
        """Get the primal residual

        Args:
            x [numpy double array], [n * 1]: [x]
            lambda_ [numpy double array], [n * 1]: [lambda]
            mu [numpy double array], [m * 1]: [mu]
            t [double]: [parameter of constraints]
            
        Returns:
            r_pri [numpy double array], [m * 1]: [the primal residual]
        """
        r_pri = np.matmul(self.A, x) - self.b 
        return r_pri
    
    def get_residue(self, x, lambda_, mu, t):
        """Get the residue

        Args:
            x [numpy double array], [n * 1]: [x]
            lambda_ [numpy double array], [n * 1]: [lambda]
            mu [numpy double array], [m * 1]: [mu]
            t [double]: [parameter of constraints]
            
        Returns:
            residue [numpy double array], [(2n + m) * 1]: [the residual]
        """
        r_dual = self.get_residual_dual(x, lambda_, mu, t) #n * 1
        r_cent = self.get_residual_center(x, lambda_, mu, t) #n * 1
        r_pri = self.get_residual_primal(x, lambda_, mu, t) #m * 1
        residue = np.concatenate((r_dual, r_cent, r_pri), axis=0)
        return residue
    
    def get_surrogate_duality_gap(self, x, lambda_, mu):
        """Get the surrogate duality gap

        Args:
            x [numpy double array], [n * 1]: [x]
            lambda_ [numpy double array], [n * 1]: [lambda]
            mu [numpy double array], [m * 1]: [mu]
            
        Returns:
            eta [double]: [the surrogate duality gap]
        """
        mask = (x > 0) * (lambda_ >= 0) 
        eta = float(np.matmul((x * mask).T, lambda_))
        return eta
        
    def get_primal_dual_direction(self, x, lambda_, mu, t, r_dual, r_cent, r_pri):
        """Get the descent direction of primal dual method

        Args:
            x [numpy double array], [n * 1]: [x]
            lambda_ [numpy double array], [n * 1]: [lambda]
            mu [numpy double array], [m * 1]: [mu]
            t [double]: [parameter of constraints]
            r_dual [numpy double array], [n * 1]: [the dual residual]
            r_cent [numpy double array], [n * 1]: [the center residual]
            r_pri [numpy double array], [m * 1]: [the primal residual]
            
        Returns:
            x_direction [numpy double array], [n * 1]: [the descent direction of x]
            lambda_direction [numpy double array], [n * 1]: [the descent direction of lambda]
            mu_direction [numpy double array], [m * 1]: [the descent direction of mu]
        """
        a11 = self.d2f(x) #n * n
        a12 = -np.identity(self.n) #n * n
        a13 = self.A.T #n * m
        a1 = np.concatenate((a11, a12, a13), axis=1) #n * (2n + m)
        a21 = np.zeros((self.n, self.n), dtype=np.float64) #n * n
        a22 = np.zeros((self.n, self.n), dtype=np.float64) #n * n
        a23 = np.zeros((self.n, self.m), dtype=np.float64) #n * m 
        for i in range(self.n):
            a21[i, i] = lambda_[i]
            a22[i, i] = x[i]
        a2 = np.concatenate((a21, a22, a23), axis=1) #n * (2n + m)
        a31 = self.A #m * n 
        a32 = np.zeros((self.m, self.n), dtype=np.float64) #m * n
        a33 = np.zeros((self.m, self.m), dtype=np.float64) #m * m 
        a3 = np.concatenate((a31, a32, a33), axis=1) #m * (2n + m)
        a = np.concatenate((a1, a2, a3), axis=0) #(2n + m) * (2n + m)
        b = np.concatenate((r_dual, r_cent, r_pri), axis=0) #(2n + m) * 1
        total_direction = -np.matmul(np.linalg.inv(a), b) #(2n + m) * 1
        x_direction = total_direction[0:self.n, :] #n * 1
        lambda_direction = total_direction[self.n:2 * self.n, :] #n * 1
        mu_direction = total_direction[2 * self.n:, :] #m * 1
        return x_direction, lambda_direction, mu_direction

    def get_learning_rate(self, x, lambda_, mu, x_direction, lambda_direction, mu_direction, t, alpha=0.4, beta=0.5):
        """Get the learning rate 

        Args:
            x [numpy double array], [n * 1]: [x]
            lambda_ [numpy double array], [n * 1]: [lambda]
            mu [numpy double array], [m * 1]: [mu]
            x_direction [numpy double array], [n * 1]: [the x descent direction]
            lambda_direction [numpy double array], [n * 1]: [the lambda descent direction]
            mu_direction [numpy double array], [m * 1]: [the mu descent direction]
            t [double]: [parameter of constraints]
            alpha (float, optional): [the alpha parameter of newton method]. Defaults to 0.4.
            beta (float, optional): [the beta parameter of newton method]. Defaults to 0.5.

        Returns:
            learning_rate [float]: [the learning rate]
        """
        up = -lambda_ 
        mask = lambda_direction < 0 
        down = lambda_direction * mask + (~mask)
        result = up / down 
        result = result * mask + (~mask)
        learning_rate = min(1, np.min(result))
        learning_rate = learning_rate * 0.99 
        while True: 
            minimal = np.min(x + x_direction * learning_rate)
            if minimal > 0:
                break 
            learning_rate = learning_rate * beta
        while True: 
            new_x = x + x_direction * learning_rate
            new_lambda = lambda_ + lambda_direction * learning_rate
            new_mu = mu + mu_direction * learning_rate
            residue_new = self.get_residue(new_x, new_lambda, new_mu, t)
            residue = self.get_residue(x, lambda_, mu, t)
            number = np.linalg.norm(residue_new, ord=2) - np.linalg.norm(residue, ord=2) * (1 - alpha * learning_rate) 
            if number <= 0:
                break 
            learning_rate = learning_rate * beta
        
        return learning_rate

    def primal_dual_method(self, x0, lambda0, mu0):
        """The primal dual method

        Args:
            x0 [numpy double array], [n * 1]: [the initial x]
            lambda0 [numpy double array], [n * 1]: [the initial lambda]
            mu0 [numpy double array], [m * 1]: [the initial mu]

        Returns:
            best_x [numpy double array], [n * 1]: [the best x]
            best_lambda [numpy double array], [n * 1]: [the best lambda]
            best_mu [numpy double array], [m * 1]: [the best mu]    
            best_y [double]: [the best y]
            log_gap_list [float array]: [the list of log(eta)]      
            log_residual_list [float array]: [the list of log(sqrt(r_dual ^ 2 + r_pri ^ 2))] 
        """
        x = x0
        lambda_ = lambda0
        mu = mu0 
        log_gap_list = []
        log_residue_list = []
        
        while True:
            fx = self.f(x)
            dfx = self.df(x)
            d2fx = self.d2f(x)
            eta = self.get_surrogate_duality_gap(x, lambda_, mu)
            t = self.n * self.u / eta
            r_dual = self.get_residual_dual(x, lambda_, mu, t)
            r_cent = self.get_residual_center(x, lambda_, mu, t)
            r_pri = self.get_residual_primal(x, lambda_, mu, t)
            norm_pri = np.linalg.norm(r_pri, ord=2)
            norm_dual = np.linalg.norm(r_dual, ord=2)
            log_gap_list.append(log(eta))
            log_residue_list.append(log(sqrt(norm_pri ** 2 + norm_dual ** 2)))
            
            if norm_dual <= self.stop_criterion and norm_pri <= self.stop_criterion and eta <= self.stop_criterion:
                best_x = x 
                best_lambda = lambda_ 
                best_mu = mu 
                best_y = fx
                break 
            
            x_direction, lambda_direction, mu_direction = self.get_primal_dual_direction(x, lambda_, mu, t, r_dual, r_cent, r_pri)
            learning_rate = self.get_learning_rate(x, lambda_, mu, x_direction, lambda_direction, mu_direction, t)
            x = x + x_direction * learning_rate
            lambda_ = lambda_ + lambda_direction * learning_rate
            mu = mu + mu_direction * learning_rate
        
        return best_x, best_lambda, best_mu, best_y, log_gap_list, log_residue_list
    
    def visualize(self, log_gap_list, log_residue_list):
        """Visualize the relationship between log(eta), log(sqrt(r_dual ^ 2 + r_pri ^ 2)) and the newton iteration times k

        Args:
            log_gap_list [float array]: [the list of log(eta)]      
            log_residue_list [float array]: [the list of log(sqrt(r_dual ^ 2 + r_pri ^ 2))] 
        """
        log_gap = np.array(log_gap_list)
        log_residue = np.array(log_residue_list)
        iter_times = len(log_gap_list)
        k = np.arange(0, iter_times)
        
        plt.xlabel('k')
        plt.ylabel('log(eta)')
        plt.plot(k, log_gap)
        plt.show()

        plt.xlabel('k')
        plt.ylabel('log(sqrt(r_dual ^ 2 + r_pri ^ 2))')
        plt.plot(k, log_residue)
        plt.show()
        
        
    def main(self):
        """The main process of primal dual method
        """     
        
        best_x, best_lambda, best_mu, best_y, log_gap_list, log_residue_list = self.primal_dual_method(self.x0, self.lambda0, self.mu0)
        save_data('primal_dual', best_x, best_lambda, best_mu, float(best_y))
        self.visualize(log_gap_list, log_residue_list)


if __name__ == "__main__":
    a = PrimalDualMethod()
    a.main()