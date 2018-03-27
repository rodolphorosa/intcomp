import sys

from numpy.random import uniform
from numpy import sign, ones, zeros, dot, mean

from utils import target
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

		iters[i], disagrees[i] = perceptron(X, y, w)
	
	return (mean(iters), mean(disagrees))

if __name__ == '__main__':
	
	if len(sys.argv) < 3: 
		print ("args: <numero de pontos> <numero de execucoes>")
		sys.exit(1)

	N = int(sys.argv[1])
	runs = int(sys.argv[2])

	print(run())