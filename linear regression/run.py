import sys
import matplotlib.pyplot as plt

from numpy import dot, mean, ones, sign, zeros
from numpy.random import uniform
from linreg import linear_regression
from evaluate import evaluate_ein, evaluate_eout
from function import linear_function
from plot import plot

def run():
	iters = zeros((runs, 1))
	ein = zeros((runs, 1))
	eout = zeros((runs, 1))

	for i in range(runs):
		
		p1 = uniform(-1, 1, 2)
		p2 = uniform(-1, 1, 2)

		f = linear_function(p1, p2)

		X = ones((n, 3))

		X[:, 1:] = uniform(-1, 1, (n, 2))

		y = sign(f(X[:, 1]) - X[:, 2])

		w = linear_regression(X, y)

		h = sign(dot(X, w.T))

		ein[i] = evaluate_ein(X, y, w, n)

		eout[i] = evaluate_eout(f, w, 1000)
	
	return mean(ein), mean(eout)

if __name__ == '__main__':
	
	if len(sys.argv) < 3:
		print("args: <numero de pontos> <numero de execucoes>")
		sys.exit(1)

	n = int(sys.argv[1])
	runs = int(sys.argv[2])

	print(run())