from math import isnan
from numpy import array, sign, dot, where, linspace
from numpy.random import choice
import matplotlib.pyplot as plt
from utils import misclassified

"""
@brief Executes perceptron learning algorithm upon a set X = [-1, 1] X [-1, 1] of N points.

"""
def perceptron(X, y, w, max_iter=10000):

	count = 0
	w1 = w

	while(count < max_iter):

		m = misclassified(X, y, w1)[0]

		if(m.size == 0):
			break

		n = choice(m)

		w1 = w1 + y[n] * X[n,:]

		count = count + 1
	
	g = sign(dot(X, w1))

	disagrees = where(y != g)

	colormap = array(['r', 'k'])	

	ymin, ymax = plt.ylim()
	xx = linspace(ymin, ymax)
	yy = (w1[1] / w1[2]) * xx - (w1[0] / w1[2])

	plt.scatter(X[:, 1], X[:, 2], c=colormap[[ 0 if i < 0 else 1 for i in g]])
	plt.plot(xx, yy, 'k-')
	plt.plot()
	plt.show()

	return (count, disagrees[0].size)