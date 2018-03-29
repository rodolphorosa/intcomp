import matplotlib.pyplot as plt

from numpy import array

"""
@brief Plots decision boundary.

"""
def plot_boundary(X, h, w, c='g'):
	colormap = ['r' if i < 0 else 'b' for i in h]
	
	slope = -(w[1]/w[2])
	intercept = -(w[0]/w[2])
	
	px = array([-1, 1])
	py = slope*px + intercept

	plt.ylim([-1, 1])
	plt.xlim([-1, 1])
	plt.scatter(X[:, 1], X[:, 2], c=colormap, marker='.')
	plt.plot(px, py, c=c, linewidth=.5, linestyle='--', label="hipothesis")

"""
@brief Plots target function
"""
def plot_target(f, c='c'):
	x = array([-1, 1])
	y = f(x)
	plt.plot(x, y, c='c', label="f(x)")

"""
@brief Exibits target function and decision boundary
"""
def plot_show():
	plt.title("Perceptron Learning Algorithm - PLA")
	plt.legend(loc="upper right")
	plt.show()
