import sys
import matplotlib.pyplot as plt

from numpy.random import uniform
from numpy import dot, mean, ones, sign, zeros
from utils import target, plot_dboundary
from perceptron import perceptron

def run():
	low, upper = -1, 1
	d = 2

	iters = zeros((runs, 1))

	for i in range(runs):
		
		p1 = uniform(low, upper, d)
		p2 = uniform(low, upper, d)		

		f = target(p1, p2)

		X = ones((N, 3))

		X[:, 1:] = uniform(low, upper, (N, d))

		y = sign(f(X[:, 1]) - X[:, 2])

		w = zeros((d + 1))

		iters[i], wt = perceptron(X, y, w)

		h = sign(dot(X, wt))

		plt.plot([-1, 1], [f(-1), f(1)], c='c', label="f(x)")
		plot_dboundary(X, h, wt, [low, upper])		
		plt.title("Perceptron Learning Algorithm - PLA")
		plt.legend(loc="upper right")
		plt.show()
	
	return mean(iters)

if __name__ == '__main__':
	
	if len(sys.argv) < 3: 
		print("args: <numero de pontos> <numero de execucoes>")
		sys.exit(1)

	N = int(sys.argv[1])
	runs = int(sys.argv[2])

	print(run())