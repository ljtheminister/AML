from scipy.io import loadmat
import numpy as np
import pandas as pd
import random
from numpy import log
from numpy import exp
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt

class LogisticSGD:

    def __init__(self, X, y, split_p, alpha, lamb, batch_size, max_iter=float('inf'), max_epoch=float('inf'), eps=.000001, step_type='constant', batch_type='constant', convergence_type='parameters'):
	self.N, self.K = X.shape
	self.X = X
	self.y = y
	self.p = split_p
	self.n_train = int(float(self.p) * self.N)
	self.n_test = self.N - self.n_train
	self.X_tilde = -y*X
	#initialize weights 0
	self.W = np.zeros(self.K)
	self.batch = batch_size

	self.alpha = alpha
	self.lamb = lamb

	self.max_iter = max_iter
	self.max_epoch = max_epoch
	self.eps = eps
	self.step_type = step_type
	self.batch_type = batch_type
	self.convergence_type = convergence_type

	row_idx = [i for i in xrange(self.N)]
	random.shuffle(row_idx)
	self.X_train = self.X_tilde[row_idx[:self.n_train], :]
	self.X_test = self.X_tilde[row_idx[self.n_train:], :]

	self.LL_iter = {}
	self.LL_mini_batches= {}
	self.LL_epoch = {}

	self.iter = 0
	self.mini_batches = 0
	self.epoch = 0

    def sigmoid(self, x):
	return 1.0/(1 + exp(-x))

    def negative_log_likelihood(self):
	self.NLL = sum(log(1 + exp(self.X_test.dot(self.W))))/float(self.n_test)

    def compute_gradient(self, x_tilde):
	if norm(self.W,2) == 0:
	    regularizer = 0
	else:
	    regularizer = self.lamb*self.W/norm(self.W,2)
	grad = x_tilde.dot(self.sigmoid(-x_tilde.dot(self.W))) + regularizer
	return grad

    def check_convergence(self, W_old=None, W_new=None, LL_old=None, LL_new=None, type='parameters'):
	if type=='parameters':
	    if max(abs(W_old - W_new)) > self.eps:
		self.converged = False
		return
	elif type=='train' or type=='test':
	    if abs(LL_old - LL_new) > self.eps:
		self.converged = False
		return
	#self.converged = True
    
    def change_step_length(self):
	if self.step_type == 'constant':
	    pass
	elif self.step_type == 'one_over_i':
	    self.alpha *= float(self.iter)**-1

	elif self.step_type == 'exp_0.6':
	    self.alpha *= float(self.iter)**-0.6

    def change_batch_length(self):
	if self.batch_type == 'constant':
	    return
	elif self.batch_type == 'linear growth':
	    self.batch += 1 
	    return

    def stochastic_gradient_descent(self):
	self.negative_log_likelihood()
	self.LL_iter[0] = self.NLL
	self.LL_mini_batches[0] = self.NLL
	self.LL_epoch[0] = self.NLL
	self.converged = False
	while not self.converged and self.iter < self.max_iter and self.epoch < self.max_epoch:
	    row_idx = [i for i in xrange(self.n_train)]
	    random.shuffle(row_idx)
	    idx = 0 
	    # Pass Thru Data
	    while idx < self.n_train:
		idx_end = idx + self.batch
		if idx_end > self.n_train:
		    idx_end = self.n_train
		# MINI-BATCHES
		# iterate through mini-batch indices and compute gradient sum
		grad_sum = np.zeros(self.K)
		while idx < idx_end:
		    grad_sum += self.compute_gradient(self.X_train[row_idx[idx]])
		    idx += 1
		    self.iter += 1
		    self.change_step_length() # altering step size based on iteration count
		    self.change_batch_length() # altering batch size based on iteration count
		self.mini_batches += 1
		W_old = self.W
		# make mini-batch gradient descent
		self.W -= self.alpha*grad_sum/float(self.batch)
		# keep track of log-likelihood after mini-batch
		self.negative_log_likelihood()
		self.LL_iter[self.iter] = self.NLL
		self.LL_mini_batches[self.mini_batches] = self.NLL
		print 'NLL: ', self.NLL
		print 'iter :', self.iter
		print 'batches: ', self.mini_batches
		# check convergence
		#self.check_convergence(W_old, self.W)
	    self.epoch += 1
	    self.LL_epoch[self.epoch] = self.NLL

    def plot_NLL_test(self, time_type='iteration'):
	if time_type == 'iteration':
	    plt.plot(self.LL_iter.keys(), self.LL_iter.values())
	    plt.xlabel('Iteration')
	    plt.ylabel('Negative Log-Likelihood')
	    plt.show()

	elif time_type == 'batch':
	    plt.plot(self.LL_mini_batches.keys(), self.LL_mini_batches.values())
	    plt.xlabel('Mini-batches')
	    plt.ylabel('Negative Log-Likelihood')
	    plt.show()


	elif time_type == 'epoch':
	    plt.plot(self.LL_epoch.keys(), self.LL_epoch.values())
	    plt.xlabel('Epoch')
	    plt.ylabel('Negative Log-Likelihood')
	    plt.show()






