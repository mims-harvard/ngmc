"""
Copyright (C) 2014  Marinka Zitnik <marinka.zitnik@fri.uni-lj.si>

This file is part of NG-MC.

NG-MC is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

NG-MC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with NG-MC.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

class Ngmc(object):
    def __init__(self, G, c, P=None, lambda_u=0.01, lambda_v=0.01, lambda_p=0.01,
                 alpha=0.1, alpha_p=0.001, max_iter=150, burnout=0,
                 callback=None):
        self.G = G
        self.Gma = np.ma.masked_array(self.G, np.isnan(self.G))
        self.P = P
        self.c = c
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_p = lambda_p
        self.alpha = alpha
        self.alpha_p = alpha_p
        self.max_iter = max_iter
        self.burnout = burnout
        self.callback = callback

        self.F = None
        self.H = None
        self.W = None

    def _initialize(self):
        self.F = np.random.rand(self.c, self.G.shape[0])
        self.H = np.random.rand(self.c, self.G.shape[1])
        if self.P:
            n_net = len(self.P)
            self.W = 1./n_net*np.ones((self.G.shape[0], n_net))

    def _g(self, x):
        return 1./(1.+np.exp(-0.5*x))

    def _g_prime(self, x):
        return 0.5*np.exp(-0.5*x)/(1.+np.exp(-0.5*x))**2

    def _F_prime(self, itr):
        G_hat = np.dot(self.F.T, self.H)
        G_hat_g_prime = self._g_prime(G_hat)
        G_hat_g = self._g(G_hat)
        der_f = np.ma.multiply(G_hat_g_prime, G_hat_g-self.Gma).T
        F_prime = np.ma.dot(self.H, der_f)
        if itr > self.burnout and self.P:
            P_multi = np.zeros(self.P[0].shape)
            for i in xrange(len(self.P)):
                Wi = self.W[:, i].reshape((self.W.shape[0], 1))
                Pi = self.P[i]
                Wit = np.tile(Wi, (1, Wi.shape[0]))
                P_multi += np.multiply(Pi, Wit)
            P2 = self.F-np.dot(self.F, P_multi.T)
            P3 = np.dot(self.F-np.dot(self.F, P_multi.T), P_multi)
            return F_prime+self.lambda_u*self.F+self.lambda_p*P2-self.lambda_p*P3
        else:
            return F_prime+self.lambda_u*self.F

    def _W_prime(self, itr):
        W_prime = np.zeros(self.W.shape)
        if itr < self.burnout:
            return W_prime
        for i in xrange(len(self.P)):
            tmp = np.dot(self.F, self.P[i].T)
            T1 = np.zeros(self.W.shape[0])
            T2 = np.zeros(self.W.shape[0])
            for u in xrange(W_prime.shape[0]):
                T1[u] = np.dot(self.F[:, u], tmp[:, u])
                T2[u] = self.W[u, i]*np.dot(tmp[:, u], tmp[:, u])

            tmp1 = np.zeros(self.F.shape)
            for p in xrange(len(self.P)):
                if p == i: continue
                tmp11 = np.dot(self.F, self.P[p].T)
                Wp = self.W[:, p].reshape((self.W.shape[0], 1))
                Wpt = np.tile(Wp, (1, self.F.shape[0])).T
                tmp1 += np.multiply(Wpt, tmp11)

            T3 = np.zeros(self.W.shape[0])
            for u in xrange(self.W.shape[0]):
                T3[u] = np.dot(tmp[:, u], tmp1[:, u])

            W_prime[:, i] = -self.lambda_p*T1+self.lambda_p*T2+self.lambda_p/2.*T3
        return W_prime

    def _H_prime(self):
        G_hat = np.dot(self.F.T, self.H)
        G_hat_g_prime = self._g_prime(G_hat)
        G_hat_g = self._g(G_hat)
        H_prime = np.ma.dot(self.F, np.ma.multiply(G_hat_g_prime, G_hat_g-self.Gma))
        return H_prime+self.lambda_v*self.H

    def fit(self, verbose=True):
        self._initialize()
        err = [1e10, 1e10]
        nrm = [1e10, 1e9]
        for itr in xrange(self.max_iter):
            if err[-1] > err[-2]: #and nrm[-1] < nrm[-2]:
                break
            err[-2] = err[-1]
            nrm[-2] = nrm[-1]
            F_prime = self._F_prime(itr)
            self.F = self.F-self.alpha*F_prime
            H_prime = self._H_prime()
            self.H = self.H-self.alpha*H_prime
            if self.P:
                W_prime = self._W_prime(itr)
                self.W = self.W-self.alpha_p*W_prime
            G_hat = self.imputed()
            sq = np.ma.multiply(self.Gma-G_hat, self.Gma-G_hat)
            fro = np.sqrt(np.nansum(sq))
            nrmse = np.sqrt(np.ma.mean(sq)/np.ma.var(self.Gma))
            if verbose:
                print "Iter: %d: Fro(G-G_hat)[known] = %5.4f" % (itr, fro)
                print "Iter: %d: NRMSE(G, G_hat)[known] = %5.4f" % (itr, nrmse)
            nrm[-1] = nrmse
            err[-1] = fro
            if self.callback:
                self.callback(self)
        return self.F, self.H, self.W

    def imputed(self):
        G_hat = np.dot(self.F.T, self.H)
        return self._g(G_hat)