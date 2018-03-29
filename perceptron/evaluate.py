from numpy import array, dot, mean, ones, sign, zeros
from misclassified import misclassified
from numpy.random import uniform

"""
@brief Returns the number of error for a given hipothesis
"""
def evaluate(f, w, n):
	X = ones((n, 3))

	X[:, 1:] = uniform(-1, 1, (n, 2))

	y = sign(f(X[:, 1]) - X[:, 2])

	h = misclassified(X, y, w)

	return h.size / n