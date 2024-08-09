import numpy as np
from numpy.linalg import svd
import math

class TFedProx_module_TNN:
    def __init__(self, args):
        self.args = args

    def converged(self, L, E, X, L_new, E_new):
        '''
        judge convered or not
        '''
        eps = self.args.eps
        condition1 = np.max(L_new - L) < eps
        condition2 = np.max(E_new - E) < eps
        condition3 = np.max(L_new + E_new - X) < eps
        return condition1 and condition2 and condition3

    def SoftShrink(self, X, tau):
        '''
        apply soft thesholding
        '''
        z = np.sign(X) * (abs(X) - tau) * ((abs(X) - tau) > 0)

        return z

    def SVDShrink(self, X, tau):
        '''
        apply tensor-SVD and soft thresholding
        '''
        W_bar = np.empty((X.shape[0], X.shape[1], 0), complex)
        D = np.fft.fft(X)
        for i in range (X.shape[2]):
            if i < X.shape[2]:
                U, S, V = svd(D[:, :, i], full_matrices = False)
                S = self.SoftShrink(S, tau)
                S = np.diag(S)
                w = np.dot(np.dot(U, S), V)
                W_bar = np.append(W_bar, w.reshape(X.shape[0], X.shape[1], 1), axis = 2)
            if i == X.shape[2]:
                W_bar = np.append(W_bar, (w.conjugate()).reshape(X.shape[0], X.shape[1], 1))
        return np.fft.ifft(W_bar).real

    def T_TSVD(self, X, epoch):
        '''
        Solve
        min (nuclear_norm(L)+lambda/(r)^[m//interval]*l2norm(E)), subject to X = L+E
        L,E
        by subgradient
        '''
        X_rs = X
        if X.ndim < 3:
            # print(X.shape)
            X_rs = np.expand_dims(X, axis=0)

        if X.ndim > 3:
            # print(X.shape)
            X_rs = np.reshape(X, (X.shape[0]*X.shape[1]*X.shape[2], X.shape[3], X.shape[-1]))
    
        m, n, l = X_rs.shape
        lamb = self.args.lamb
        max_iters = 1000
        L = np.zeros((m, n, l), float)
        E = np.zeros((m, n, l), float)
        Y = np.zeros((m, n, l), float)
        if self.args.verbose:
            print('The initial truncation threshold: ', 1 / (2 * lamb))
            print('Current truncation threshold is: ', 1 / (2 * lamb) * math.pow(self.args.r, math.ceil(epoch / self.args.interval)))
        iters = 0
        while True:
            iters += 1
            # update L
            L_new = self.SVDShrink(X_rs, 1 / (2 * lamb) * math.pow(self.args.r, math.ceil(epoch / self.args.interval)))
            # update E
            E_new = X_rs - L_new
            if self.converged(L, E, X_rs, L_new, E_new) or iters >= max_iters:
                return L_new.reshape(X.shape), E_new.reshape(X.shape)
            else:
                L, E = L_new, E_new
                if self.args.verbose:
                    print(np.max(L_new - L))