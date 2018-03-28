from math import isnan
from numpy import array, sign, dot, where, linspace
from numpy.random import choice
from utils import misclassified, plot_dboundary

"""
@brief Executes perceptron learning algorithm upon a set X = [-1, 1] X [-1, 1] of N points.

"""
def perceptron(X, y, w, max_iter=10000):

	t = 0
	wt = w

	while(t < max_iter):

		m = misclassified(X, y, wt)

		if(m.size == 0):
			break

		n = choice(m)

		wt = wt + ( X[n,:] * y[n] )

		t = t + 1

	return (t, wt)