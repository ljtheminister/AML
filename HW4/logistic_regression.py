''' Logistic regression using stochastic gradient descent with options for step length: pre-specified constant proportional to 1/k proportional to k^(0.6)
'''

from scipy.io import loadmat
import numpy as np
import pandas as pd
import random
from numpy import log
from numpy import exp
from numpy import dot
from numpy.linalg import norm

data = loadmat('covtype.mat')
y = data['y']
X = data['X']
N, K  = X.shape
X_tilde = compute_X_tilde(X, y)
X_train, X_test = data_split(X_tilde, p, N, K) #X_train and X_test are X_tilde split into test and training sets

#class LogisticRegression:

def data_split(X, p, N, K):
    # check to see that 0 < p < 1
    n_train = int(float(p) * N)
    row_idx = random.shuffle([i for i in xrange(n_train)])
    X_train = X[row_idx[:n_train], :]
    X_test = X[row_idx[n_train:], :]
    return X_train, X_test

def sigmoid(x):
    return 1.0/(1 + exp(-x))

def compute_X_tilde(X, y):
    return -y*X

def SGD_step(x_tilde, theta, lamb):
    grad = np.zeros(K)
    LL = 0
    grad += x_tilde.dot(sigmoid(-x_tilde.dot(theta))) + lamb
    LL += log(1 + exp(x_tilde.dot(theta))) + lamb*norm(theta,2)
    return grad, LL

def SGD(X, theta, lamb, alpha, N, K):
    row_idx = random.shuffle([i for i in xrange(N)])
    for idx in row_idx:
	grad, LL = SGD_step(X[idx,:], theta, lamb)

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

