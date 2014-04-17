'''
Logistic regression using stochastic gradient descent with options for step length:
pre-specified constant
proportional to 1/k
proportional to k^(0.6)
'''

import numpy as np
import pandas as pd
import random
from numpy import log
from numpy import exp
from numpy import dot
from numpy.linalg import norm

#class LogisticRegression:

N, K  = X.shape

def data_split(X, p, N, K):
    # check to see that 0 < p < 1
    n_train = int(float(p) * N)
    row_idx = random.shuffle([i for i in xrange(n_train)])
    X_train = X.ix[row_idx[:n_train], :]
    X_test = X.ix[row_idx[n_train:], :]
    return X_train, X_test

def sigmoid(x):
    return 1.0/(1 + exp(-x))

def X_tilde(X, y):
    return -X.dot(y)


def SGD(X_tilde, theta, lamb, alpha, N, K):
    grad = np.zeros(K)
    LL = 0
    grad += X_tilde.dot(sigmoid(-X_tilde.dot(theta))) + lamb
    LL += log(1 + exp(X_tilde.dot(theta)) + lamb*norm(theta,2)
    return alpha*grad/float(N), LL/float(N)

def not_converged(a, b, eps):
    if max(abs(a-b)) > eps:
	return True
    return False


def main():
# randomly initialize thetas
theta_old = np.zeros(K)
LL_list = list()

while not_converged(theta_old, theta):
    grad, LL = SGD(X_tilde, theta_old, lamb, alpha, N, K)
    theta = theta_old + grad

