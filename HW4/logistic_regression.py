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

class LogisticRegression:

    def __init__(self, X, y, split_p, alpha, lamb, batch_size, max_iter=float('inf'), max_epoch=float('inf')):
	self.N, self.K = X.shape
	self.X = X
	self.y = y
	self.p = split_p
	self.X_tilde = -y*X
	#initialize weights 0
	self.W = np.zeros(self.K)
	self.batch = batch_size

	self.alpha = alpha
	self.lamb = lamb
	self.max_iter = max_iter
	self.max_epoch = max_epoch

	n_train = int(float(p) * N)
	row_idx = [i for i in xrange(N)]
	random.shuffle(row_idx)
	self.X_train = X_tilde[row_idx[:n_train], :]
	self.X_test = X_tilde[row_idx[n_train:], :]

	self.LL_iter = {}
	self.LL_epoch = {}


    def sigmoid(x):
	return 1.0/(1 + exp(-x))

    def log_likelihood(self):
	LL = sum(log(1 + exp(self.X_tilde.dot(self.W))))/float(self.N) + self.lamb*norm(self.W, 2)
	return LL

    def compute_gradient(self, x_tilde):
	grad = x_tilde.dot(sigmoid(-x_tilde.dot(self.W))) + lamb
	return grad


    def stochastic_gradient_descent(self):

	LL = log_likelihood(self)
	self.LL_iter[0] = LL
	self.LL_epoch[0] = LL

	iter = 0
	epoch = 0

	while iter < self.max_iter and epoch < max_epoch:

	    row_idx = [i for i in xrange(self.N)]
	    random.shuffle(row_idx)
















	return LL_iter, LL_epoch





'''
def data_split(X, p, N, K):
    # check to see that 0 < p < 1
    n_train = int(float(p) * N)
    row_idx = [i for i in xrange(N)]
    random.shuffle(row_idx)
    X_train = X[row_idx[:n_train], :]
    X_test = X[row_idx[n_train:], :]
    return X_train, X_test


def compute_X_tilde(X, y):
    return -y*X
'''



def SGD_step(self, x_tilde):
    
def 




def SGD(X, theta, lamb, alpha, N, K, iter):
    LL_iter = {}
    row_idx = [i for i in xrange(N)]
    random.shuffle(row_idx)
    for idx in row_idx:
	if not_converged(LL_iter[iter-1], LL_iter[iter])
	    grad, LL = SGD_step(X[idx,:], theta, lamb)
	    theta -= alpha*grad
	    iter += 1 #increment iteration counter
	    print 'Log Likelihood: ', LL
	    LL_iter[iter] = LL
	else:
	    return theta, iter, passes
    passes += 1 #increment full data pass counter
    LL_passes[passes] = LL

def SGD_batch(X, theta, lamb, indices):



# BATCH SIZE
def SGD(X, theta, lamb, alpha, N, K, batch_size, eps):
    LL_iter = {}
    LL_pass = {}


    iter = 0
    passes = 0

    theta = np.zeros(K)



    row_idx = [i for i in xrange(N)]
    random.shuffle(row_idx)
    idx = 0











def SGD(X, theta, lamb, alpha, N, K, passes):
    LL_iter = {}
    row_idx = [i for i in xrange(N)]
    random.shuffle(row_idx)
    for idx in row_idx:
	if not_converged(LL_iter[iter-1], LL_iter[iter])
	    grad, LL = SGD_step(X[idx,:], theta, lamb)
	    theta -= alpha*grad
	    iter += 1 #increment iteration counter
	    print 'Log Likelihood: ', LL
	    LL_iter[iter] = LL
	else:
	    return theta, iter, passes
    passes += 1 #increment full data pass counter
    LL_passes[passes] = LL





def not_converged(a, b, eps):
    if max(abs(a-b)) > eps:
	return True
    return False


def logistic_regression_SGD(X, y, 
# randomly initialize thetas
    theta_old = np.zeros(K)
    LL_list = list()

while not_converged(theta_old, theta):
    grad, LL = SGD(X_tilde, theta_old, lamb, alpha, N, K)
    theta = theta_old + grad


if __name__ = '__main__':
data = loadmat('covtype.mat')
y = data['y']
X = data['X']
N, K  = X.shape
p = .9 # 90-10 train-test split
X_tilde = compute_X_tilde(X, y)
X_train, X_test = data_split(X_tilde, p, N, K) #X_train and X_test are X_tilde split into test and training sets
lamb = .01


