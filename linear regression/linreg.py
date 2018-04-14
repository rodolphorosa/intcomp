from numpy import dot
from numpy.linalg import pinv

"""
@brief Executes linear regression algorithm upon a set X = [-1, 1] X [-1, 1] of N points.

"""
def linear_regression(X, y):
	return dot(pinv(X), y)