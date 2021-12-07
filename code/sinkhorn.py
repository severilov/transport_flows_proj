import warnings
import copy
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
np.set_printoptions(suppress=True)


class Sinkhorn:
    """
    Корректную реализация Синхорна. Совпадает с псевдокодом алгоритма 2 
    из https://arxiv.org/pdf/2005.11604.pdf. Замечания три:
    1) не обнаружено параметра gamma (видимо либо он уже зашит в косты, либо
    его принимают равным 1;
    2) под lambda_L, lambda_W понимаются не те переменные, что в псевдокоде алгоритма.
    Чтобы получить точное соответствие, необходимо переписать формулы там для оригинальных
    переменных lambda_W, lambda_L, их переобозначение происходит сверху страницы 9.
    3) не совсем ясно, что возвращается в методе rec_d_i_j. Вроде выглядит похожим на 
    матрицу d_i_j из статьи, однако почему-то отсутствует нормализующий знаменатель.
    Также почему-то результат умножают на общее число людей. Судя по тому, что авторы 
    кода называют результат не d_i_j, а r, и метод называется rec_d_i_j, имеет значение, 
    немного отличающееся от d_i_j из статьи.
    """
    def __init__(self, L, W, people_num, iter_num, eps):
        """
        Initialize all parameters for Sinkhorn.
        ----------
        Arguments:
            L, W: np.array
                Should be 1-dimensional and both have equal lengths.
            people_num: int
                Total people num, needed for d_i_j matrix computation.
            iter_num: int
                Number of iterations of algorithm.
            eps: float
        """
        self.L = L
        self.W = W
        assert (len(L) == len(W))
        self.n = len(L)
        self.people_num = people_num
        self.num_iter = iter_num
        self.eps = eps
        self.multistage_i = 0

    def sinkhorn(self, k, cost_matrix, lambda_W_prev, lambda_L_prev):
        """
        Single iteration of Sinkhorn algorithm. In fact, this is algorithm 2 of 
        https://arxiv.org/pdf/2005.11604.pdf, written for original, not redefined
        variables lambda_W, lambda_L. Gamma parameter from algorithm seems 
        to be either integrated into cost_matrix or equal 1.
        Takes as input lambda's values on previous iteration, cost_matrix and iteration number.
        Returns updated values of lambda_L and lambda_W.
        ----------
        Arguments:
            k: int
                number of iteration
            cost_matrix: np.array
                T_ij from paper, its shape is (n x n).
            lambda_W_prev: np.array
                Should be 1-dimensional and have lengths of n.
        ----------
        Returns:
            lambda_L, lambda_W: np.array
                Are 1-dimensional and have lengths n.
        """
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
        """
        Perform Sinkhorn iteration process on cost_matrix.
        Start points are zero vectors.
        Receives as input cost_matrix T_ij from paper.
        Returns lambda_L, lambda_W and some analogue of d_i_j matrix from paper.
        ----------
        Arguments:
            cost_matrix: np.array
                Should be 2-dimensional and have shape (n x n)
        ----------
        Returns:
            r: np.array
                Is 2-dimensional and has shape (n x n), containing matrix d(i, j).
            lambda_L, lambda_W: np.array
                Are 1-dimensional and have lengths n.
        """
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
        """
        Calculate some analogue of matrix d_i_j from paper. Not very clear why, but
        here we don't have normalizing denominator (sum of matrix values). Also
        for some purpose matrix is multiplied by total number of people.
        ----------
        Arguments:
            lambda_L, lambda_W: np.array
                Should be 1-dimensional and have lengths of n.
            cost_matrix: np.ndarray
                Should be 2-dimensional and have shape (n x n)
        ----------
        Returns:
            d_i_j: np.array
                Is 2-dimensional and has shape (n x n), containing matrix d(i, j).
        """
        er = np.exp(-1 - cost_matrix - (np.reshape(lambda_L, (self.n, 1)) + lambda_W))
        d_i_j = er * self.people_num
        return d_i_j


class AcceleratedSinkhorn:
    """
    Максимально близкая к псевдокоду статьи, и потому не очень оптимальная реализация
    ускоренного Синхорна. Для лучшего понимания рекомендую одновременно открыть 
    псевдокод алгоритма 3 из https://arxiv.org/pdf/2005.11604.pdf и
    псевдокод алгоритма 4 из https://arxiv.org/pdf/1906.03622.pdf (он в Supplementary).
    Надеюсь, что нигде не накосячил с переобозначенными лямбда_W и лямбда_L.
    Проблемы и замечания:
    1) Я все еще хз, что делать с people_num и откуда оно взялось в реализации обычного
    Синхорна.
    2) Также все еще открыт вопрос с нормировкой, а точнее её отсутствием в оригинальном
    Синхорне. Тут все сделано с ней, но как бы...

    TODO: not forget about converting lambda_'s back to their original meaning.
    """
    def __init__(self, l, w, T_ij, people_num, steps, eps_f, eps_eq):
        """
        Initialize AcceleratedSinkhorn algorithm instance. Unlike implementation
        of original algortihm, T_ij is removed from arguments of methods and 
        implemented as attribute of instance.
        ----------
        Arguments:
            l, w: np.array
                Should be 1-dimensional and have lengths of n.
            T_ij: np.ndarray
                Should be 2-dimensional and have shape (n x n).
            people_num: int
            steps: int
            eps_f, eps_eq: float
        """
        self.l = l
        self.w = w
        self.T_ij = T_ij
        assert (len(l) == len(w))
        self.n = len(l)
        self.people_num = people_num
        self.steps = steps
        self.eps_f = eps_f
        self.eps_q = eps_eq

    def sinkhorn_step(self, k, y_l, y_w):
        """
        Here is version of Sinkhorn iteration for redefined lambda's (here they are
        called as y_l, y_w). It could be optimized, but I intentionally do not do that to
        achieve the best possible clarity of code.
        """
        if k % 2 == 0:
            y_l_new = y_l + np.log(self.l) - np.log(self.B_ij(y_l, y_w).sum(axis=1))
            y_w_new = y_w
        else:
            y_l_new = y_l
            y_w_new = y_w + np.log(self.w) - np.log(self.B_ij(y_l, y_w).sum(axis=0))
        return y_l_new, y_w_new

    def B_ij(self, lambda_l, lambda_w):
        """
        Compute matrix B_ij from lambdas. Lambdas are assumed to be already redefined 
        as in top of the page 9 in russian paper.
        ----------
        Arguments:
            lambda_l, lambda_w: np.array
                Should be 1-dimensional and have lengths of n.
        ----------
        Returns:
            B_ij: np.ndarray
                Is 2-dimensional and has shape (n x n), containing matrix B(i, j).
        """
        B_ij = np.exp(-self.T_ij + lambda_l.reshape((self.n, 1)) + lambda_w)
        return B_ij

    def f(self, d_ij):
        """
        f-function from paper. Gamma assumed to be 1 (or already taken into 
        account in T_ij).
        ----------
        Arguments:
            d_ij: np.ndarray
                Should be 2-dimensional and have shape (n x n).
        ----------
        Returns:
            f: float
        """
        f = np.sum(self.T_ij * d_ij) + np.sum(d_ij * np.log(d_ij))
        return f

    def phi(self, lambda_l, lambda_w):
        """
        Phi-function from paper. Lambdas are assumed to be already redefined 
        as in top of the page 9 in russian paper.
        ----------
        Arguments:
            lambda_l, lambda_w: np.arrays
                Should be 1-dimensional and have lengths of n.
        ----------
        Returns:
            phi: float
        """
        phi = np.log(self.B_ij(lambda_l, lambda_w).sum()) - \
                (lambda_l * self.l).sum() - (lambda_w * self.w).sum()
        return phi

    def grad_phi(self, lambda_l, lambda_w):
        """
        Grad of phi-function from paper. Lambdas are assumed to be already redefined 
        as in top of the page 9 in russian paper.
        ----------
        Arguments:
            lambda_l, lambda_w: np.arrays
                Should be 1-dimensional and have lengths of n.
        ----------
        Returns:
            grads: list
                Contains two 1-dimensional np.arrays of length n.
        """
        B = self.B_ij(lambda_l, lambda_w)
        B_part = B.sum(axis=1) / B.sum() # (B(l, w) * 1) / (1 * B(l, w) * 1)
        grads = [-self.l + B_part, -self.w + B_part]
        return grads

    def d_ij(self, lambda_l, lambda_w):
        """
        Compute matrix d_ij from lambdas. Lambdas are assumed to be already redefined 
        as in top of the page 9 in russian paper.
        ----------
        Arguments:
            lambda_l, lambda_w: np.array
                Should be 1-dimensional and have lengths of n.
        ----------
        Returns:
            B_ij: np.ndarray
                Is 2-dimensional and has shape (n x n), containing matrix B(i, j).
        """
        B = self.B_ij(lambda_l, lambda_w)
        d_ij = B / B.sum() # # (B(l, w) * 1) / (1 * B(l, w) * 1)
        return d_ij

    def step(self, L_k, a_k, v_k, x_k, d_hat_k):
        """
        Single iteration of accelerated Sinkhorn. Lambdas are assumed to be already redefined 
        as in top of the page 9 in russian paper.
        All variables fully correspond to those in paper.
        ----------
        Arguments:
            L_k, a_k: float
            v_k, x_k: lists
                Both should have two 1-dimensional np.arrays of lengths n.
            d_hat_k: np.ndarray
                Should be 2-dimensional and have shape (n x n).
        ----------
        Returns:
            L: float
            a: float
            v: list
                Contains two 1-dimensional np.arrays of length n.
            x: list
                Contains two 1-dimensional np.arrays of length n.
            d_hat: np.ndarray
                Is 2-dimensional and has shape (n x n), containing matrix d_hat(i, j).
        """
        L = {'k': L_k, 'k+1': None}
        a = {'k': a_k, 'k+1': None}
        v = {'k': v_k, 'k+1': [None, None]}
        x = {'k': x_k, 'k+1': [None, None]}
        d_hat = {'k': d_hat_k, 'k+1': None}
        tau = {'k': None}
        y = {'k': [None, None]}
        i = {'k': None}
        d = {'k': None}
        

        L['k+1'] = L['k'] / 2
        while True:
            a['k+1'] = 1/(2*L['k+1']) + \
                    ( 1/(4*L['k+1']**2) + a['k']**2 * L['k']/L['k+1'] )**0.5

            tau['k'] = 1/(a['k+1']*L['k+1'])

            # x['k'] and v['k'] are lists so y['k'] should be computed
            # by applying according function to them component wise
            y_func = lambda v, x: tau['k']*v + (1-tau['k'])*x 
            y['k'] = list(map(y_func, zip(v['k'], x['k'])))

            # now, compute gradient analytically
            phi_grad = self.grad_phi(*y['k'])
            phi_grad_norms = [np.square(grad).sum() for grad in phi_grad]
            i['k'] = np.argmax(phi_grad_norms)

            # perform sinkhorn step (that 'if' block in the middle of algorithm)
            x['k+1'][0], x['k+1'][1] = self.sinkhorn_step(i['k'], *y['k'])

            # do the same thing as when updating y['k'] to v['k']
            v_func = lambda v, grad: v - a['k+1']*grad
            v['k+1'] = list(map(v_func, zip(v['k'], phi_grad)))
            
            # check inner loop break condition
            cond = self.phi(*x['k+1']) <= self.phi(*y['k+1']) + sum(phi_grad_norms)/(2*L['k+1'])
            if cond:
                d['k'] = self.d_ij(y['k'])
                d_hat['k+1'] = a['k+1']*d['k'] + L['k']*(a['k']**2)*d_hat['k'] / \
                                        ( L['k+1']*(a['k+1']**2) )
                break
            L['k+1'] = 2*L['k+1']
        
        return L['k+1'], a['k+1'], v['k+1'], x['k+1'], d_hat['k+1']
    
    def iterate(self, x_0=None, L_0=1, a_0=0):
        """
        Iteration process of accelerated Sinkhorn from paper.
        x_0 is starting point.
        L_0 is initial estimate of Lipschitz constant.
        a_0 is God knows what
        ----------
        Arguments:
            x_0: list
                Should have two 1-dimensional np.arrays of length n.
                If None, default version of two zero vectors is used.
            L_0, a_0: float
        ----------
        Returns:
            d_hat: np.ndarray
                2-dimensional np.ndarray of shape (n x n), containing matrix d_hat(i, j).
            x: list
                Contains two 1-dimensional np.arrays of length n.
        """
        L = L_0
        if x_0 is None:
            x_0 = [np.zeros(self.n), np.zeros(self.n)]
        else:
            x = copy.deepcopy(x_0)
        a = a_0
        v = copy.deepcopy(x)
        d_hat = self.d_ij(x_0)
        k = 0

        while not self.criterion(d_hat, x) or (k <= self.steps):
            L, a, v, x, d_hat = self.step(L, a, v, x, d_hat)
        
        return d_hat, x

    def criterion(self, d_ij, x, eps_f, eps_eq) -> bool:
        """
        Stop criterion for accelerated Sinkhorn from paper.
        ----------
        Arguments:
            d_ij: np.ndarray
                Should be 2-dimensional and have shape (n x n).
            x: list
                Should have two 1-dimensional np.arrays of length n.
            eps_f, eps_eq: float
        ----------
        Returns:
            bool
        """
        phi = self.phi(*x)
        f = self.f(d_ij)

        first = np.abs(phi + f) < eps_f
        second = np.linalg.norm(d_ij.sum(axis=1) - self.l) < eps_eq
        third = np.linalg.norm(d_ij.sum(axis=0) - self.w) < eps_eq

        return first and second and third

    def redefine_lambdas(lambda_l, lambda_w):
        """
        Perform redefining lambdas as on top of page 9.
        Interesting fact, this transformation is inverse to itself.
        I mean, f(f(x)) = x. So to get original values, just apply
        this function once again to transformed values.
        Заскамили алгоритм на обратную функцию...
        ----------
        Arguments:
            lambda_l, lambda_w: np.array
                Should be 1-dimensional.
        ----------
        Returns:
            new_lambda_l, new_lambda_w: np.array
                Are 1-dimensional and have same lengths as original.
        """
        new_lambda_l = -lambda_l - 1/2
        new_lambda_w = -lambda_w - 1/2
        return new_lambda_l, new_lambda_w

