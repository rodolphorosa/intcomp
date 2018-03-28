from numpy import where, sign, dot, array
import matplotlib.pyplot as plt

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
	return where(sign(dot(X, w)) != y)[0]

"""
@brief Plots decision boundary.

"""
def plot_dboundary(X, h, w):
	colormap = ['r' if i < 0 else 'b' for i in h]	
	
	slope = - (w[1] / w[2])	
	intercept = - (w[0] / w[2])
	
	px = array([[-1, 0], [1, 0]])	
	py = slope * px + intercept

	plt.scatter(X[:, 1], X[:, 2], c=colormap, marker='+')
	plt.plot(px, py, c='g', linewidth=.5, linestyle='--')
	plt.ylim([-1, 1])
	plt.show()