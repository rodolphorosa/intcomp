import sys
import matplotlib.pyplot as plt

from plot import plot_boundary, plot_target, plot_show
from numpy import array, dot, mean, ones, sign, zeros
from perceptron import perceptron
from numpy.random import uniform
from function import target
from evaluate import evaluate

def run():
	iters = zeros((runs, 1))
	diffs = zeros((runs, 1))

	for i in range(runs):
		
		p1 = uniform(-1, 1, 2)
		p2 = uniform(-1, 1, 2)

		f = target(p1, p2)

		X = ones((N, 3))

		X[:, 1:] = uniform(-1, 1, (N, 2))

		y = sign(f(X[:, 1]) - X[:, 2])

		w = zeros(3)

		iters[i], wt = perceptron(X, y, w)
		
		diffs[i] = evaluate(f, wt, 1000)
	
	return (mean(iters), mean(diffs))

if __name__ == '__main__':
	
	if len(sys.argv) < 3:
		print("args: <numero de pontos> <numero de execucoes>")
		sys.exit(1)

	N = int(sys.argv[1])
	runs = int(sys.argv[2])

	print(run())