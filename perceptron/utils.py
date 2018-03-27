from numpy import where, sign, dot

"""
@brief Returns the equation of the line passing through points p1 and p2.

"""

def target(p1, p2):
	m = (p2[1] - p1[1]) / (p2[0] - p1[0])
	b = p1[1] - m * p1[0]
	return lambda x : m * x + b

"""
@brief Returns the indexes of misclassified points. 

"""
def misclassified(X, y, w):
	return where(sign(dot(X, w)) != y)