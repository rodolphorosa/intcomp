import sys

from numpy.random import uniform
from numpy import sign, ones, zeros, dot, mean

from utils import target, plot_dboundary
from perceptron import perceptron

def run():
	low, upper = -1, 1
	d = 2

	iters = [0] * runs
	disagrees = [0] * runs

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

	plot_dboundary(X, h, wt)
	
	return mean(iters)

if __name__ == '__main__':
	
	if len(sys.argv) < 3: 
		print ("args: <numero de pontos> <numero de execucoes>")
		sys.exit(1)

	N = int(sys.argv[1])
	runs = int(sys.argv[2])

	print(run())