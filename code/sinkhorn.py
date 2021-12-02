import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
np.set_printoptions(suppress=True)


class Sinkhorn:

    def __init__(self, L, W, people_num, iter_num, eps):
        self.L = L
        self.W = W
        assert (len(L) == len(W))
        self.n = len(L)
        self.people_num = people_num
        self.num_iter = iter_num
        self.eps = eps
        self.multistage_i = 0

        self.prev_parameters = {}
        self.temp_parameters = {}

    def sinkhorn(self, k, cost_matrix, lambda_W_prev, lambda_L_prev):

        if k % 2 == 0:
            lambda_W = lambda_W_prev
            lambda_L = np.log(np.nansum(
                (np.exp(-lambda_W_prev - 1 - cost_matrix)).T
                / self.L, axis=0
            ))
        else:
            lambda_L = lambda_L_prev
            lambda_W = np.log(np.nansum(
                (np.exp(-lambda_L - 1 - cost_matrix.T)).T
                / self.W, axis=0
            ))
        return lambda_W, lambda_L

    def iterate(self, cost_matrix):
        cost_matrix[cost_matrix == 0.0] = 100.0

        lambda_L = np.zeros(self.n)
        lambda_W = np.zeros(self.n)

        for k in range(self.num_iter):

            lambda_Wn, lambda_Ln = self.sinkhorn(k, cost_matrix, lambda_W, lambda_L)

            delta = np.linalg.norm(np.concatenate((lambda_Ln - lambda_L,
                                                   lambda_Wn - lambda_W)))

            lambda_L, lambda_W = lambda_Ln, lambda_Wn

            if delta < self.eps:
                print(f"number of iterations in Sinkhorn:{k}")
                break
        r = self.rec_d_i_j(lambda_Ln, lambda_Wn, cost_matrix)
        return r, lambda_L, lambda_W

    def rec_d_i_j(self, lambda_L, lambda_W, cost_matrix):
        er = np.exp(-1 - cost_matrix - (np.reshape(lambda_L, (self.n, 1)) + lambda_W))
        return er * self.people_num

    ############## here starts updates

    def set_temp_prev(self):
        for key in ['a', 'A', 'eta', 'theta']:
            self.prev_parameters[key] = 0
            self.prev_parameters['L'] = 1
            self.temp_parameters['L'] = 1

    def update_prev(self):
        self.prev_parameters.update(self.temp_parameters)

    def accelerated(self, k, cost_matrix, lambda_W_prev, lambda_L_prev):
        get_B = lambda lmbd, mu: [[-lmbd[i] - mu[j] - cost_matrix[i, j] for i in range(len(lmbd))]
        for j in range(len(mu)) ]

    @staticmethod
    def right(B):
        return np.nansum(B, axis=0)

    @staticmethod
    def left(B):
        return np.nansum(B.T, axis=0)

    def count(self, B):
        r = self.right(B)
        l = self.left(B)
        S = np.nansum(r)
        return r, l, S

    def grad(self, B, r=None, l=None, s=None):
        returns = []
        if r is None:
            r = self.right(B)
        returns = [r]
        if l is None:
            l = self.left(B)
        returns = returns + [l]
        if s is None:
            S = np.nansum(r)
        returns = returns + [S]
        g = [-self.L + r/S, -self.W + l/S]
        return tuple([g] +returns)

    def phi(self, Bsum, lmbd, mu):
        return np.log(Bsum) - lmbd @ self.L - mu @ self.W

        while True:
            self.temp_parameters['L'] = self.temp_parameters['L']/2
            L = self.temp_parameters['L']
            self.temp_parameters['a'] = 1/(2*L) + (1/(4*L**2) + self.prev_parameters['a']**2*self.prev_parameters['L']/L)**0.5
            a = self.temp_parameters['a']
            tau = 1/(a*L)
            lmbd = [tau*self.prev_parameters['eta'][i] + (1 - tau)*[lambda_L_prev, lambda_W_prev][i] for i in [0, 1]]
            B = self.get_B(*lmbd)
            g, r, l, _ = self.grad(B)
            g2 = g**2
            i = np.argmax(g2)
            lambda_L, lambda_W = self.sinkhorn(i, cost_matrix, lambda_W_prev, lambda_L_prev)
            self.temp_parameters['eta'] = self.prev_parameters['eta'] - a*g
            B_res = get_B(lambda_L, lambda_W)
            phi_res = phi(B_res, lambda_L, lambda_W)
            if phi_res <= phi(B, *lmbd) - g2/(2*L):
                break

        self.temp_parameters['L'] = self.temp_parameters['L']/2
        self.update_prev()
        return lambda_L, lambda_W