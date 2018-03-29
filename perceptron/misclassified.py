from numpy import dot, sign, where

"""
@brief Returns the indexes of misclassified points. 

"""
def misclassified(X, y, w):
	return where(sign(dot(X, w.T)) != y)[0]