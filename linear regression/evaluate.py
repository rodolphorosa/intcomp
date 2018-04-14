from numpy import array, dot, mean, ones, sign, where, zeros
from numpy.random import uniform

"""
@brief Returns the indeces of misclassified points for a given hipothesis
"""
def misclassified(X, y, w):
	h = sign(dot(X, w))
	
	return where(h != y)[0].size

"""
@brief Returns the percentage of out-of-sample errors for a given hipothesis
"""
def evaluate_eout(f, w, n):
	X = ones((n, 3))

	X[:, 1:] = uniform(-1, 1, (n, 2))

	y = sign(f(X[:, 1]) - X[:, 2])

	errors = misclassified(X, y, w)

	return errors / n

"""
@brief Returns the percentage of in-sample errors for a given hipothesis
"""
def evaluate_ein(X, y, w, n):
	errors = misclassified(X, y, w)
	
	return errors / n