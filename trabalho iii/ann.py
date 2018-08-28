import numpy as np
import matplotlib.pyplot as plt
from common import *
from plot import *

import time

class Network:
    def __init__(self, sizes):
        model = {}
        model['l1'] = np.random.uniform(-1, 1, (sizes[1], sizes[0]))
        model['l2'] = np.random.uniform(-1, 1, (sizes[2], sizes[1]))
        model['b1'] = np.random.uniform(-1, 1, sizes[1])
        model['b2'] = np.random.uniform(-1, 1, sizes[2])
        self.model = model

    def fit(self, X, y):
        self.sgd(X, y)

    def feedforward(self, x):
        hl = tanh(self.model['l1'] @ x + self.model['b1'])
        return tanh(self.model['l2'] @ hl + self.model['b2'])

    def sgd(self, X, y, epochs=100, eta=0.1, lamb=0.0001, mode="mini-batch", test=False):
        if (test):
            train_set, test_set = divide_train_test(len(X), .1)
            X_test = X[test_set]
            y_test = y[test_set]
            X = X[train_set]
            y = y[train_set]

        n = len(X)
        decay = 1 - (2 * eta * (lamb/n))

        if (mode=="online"):
            minibatch_size = 1
        elif (mode=="mini-batch"):
            minibatch_size = n//100
        elif (mode=="batch"):
            minibatch_size = n

        ein = np.zeros(epochs)
        eout = np.zeros(epochs)
        for e in range(epochs):
            perm = np.random.permutation(n)
            Xs = X[perm]
            ys = y[perm]

            for i in range(0, n, minibatch_size):
                X_batch = X[i:i+minibatch_size]
                y_batch = y[i:i+minibatch_size]
                self.update_weights(X_batch, y_batch, eta, decay)
            if (test):
                ein[e] = self.regularized_cost(X, y, lamb)
                eout[e] = self.regularized_cost(X_test, y_test, lamb)
                print("epoca {0}: ein {1}, eout {2}".format(e, ein[e], eout[e]))
            else:
                print("Epoca", e)
        if (test):
            plt.plot(np.arange(epochs), ein, 'b', np.arange(epochs), eout, 'g')
            plt.legend(("treino", "teste"))
            plt.xlabel("epocas")
            plt.ylabel("função de custo")
            plt.show()

    def update_weights(self, X, y, eta, decay):
        n = len(X)
        nablas = {}
        nablas['l1'] = np.zeros(self.model['l1'].shape)
        nablas['b1'] = np.zeros(self.model['b1'].shape)
        nablas['l2'] = np.zeros(self.model['l2'].shape)
        nablas['b2'] = np.zeros(self.model['b2'].shape)

        for xi, yi in zip(X, y):
            grads = self.backpropagate(xi, yi)
            nablas['l1'] = nablas['l1'] + grads['l1']
            nablas['l2'] = nablas['l2'] + grads['l2']
            nablas['b1'] = nablas['b1'] + grads['b1']
            nablas['b2'] = nablas['b2'] + grads['b2']

        self.model['l1'] = decay * self.model['l1'] - (eta/n)*nablas['l1']
        self.model['l2'] = decay * self.model['l2'] - (eta/n)*nablas['l2']
        self.model['b1'] = self.model['b1'] - (eta/n)*nablas['b1']
        self.model['b2'] = self.model['b2'] - (eta/n)*nablas['b2']

    def backpropagate(self, x, y):
        z1 = self.model['l1'] @ x + self.model['b1']
        hl = tanh(z1)
        z2 = self.model['l2'] @ hl + self.model['b2']
        ol = tanh(z2)

        nablas = {}
        delta = self.cost_derivative(ol, y) * tanh_derivative(z2)
        nablas['l2'] = np.vstack(delta) @ np.vstack(hl).T
        nablas['b2'] = delta

        delta_h = (self.model['l2'].T @ delta) * tanh_derivative(z1)
        nablas['l1'] = np.vstack(delta_h) @ np.vstack(x).T
        nablas['b1'] = delta_h
        return nablas

    def predict(self, X):
        return sign(self.decision_function(X))

    def decision_function(self, X):
        n = len(X)
        y = np.zeros(n)
        for i in range(n):
            y[i] = self.feedforward(X[i])
        return y

    def score(self, X, y):
        yy = self.predict(X)
        return np.mean(yy == y)

    def cost(self, X, y):
        return np.mean((self.decision_function(X) - y)**2)

    def regularized_cost(self, X, y, lamb):
        n = len(X)
        wTw = 0
        for w in ['l1', 'b1', 'l2', 'b2']:
            wTw = wTw + np.sum(self.model[w]**2)
        return self.cost(X, y) + ((lamb/n)*wTw)

    def cost_derivative(self, output, target):
        return 2 * (output - target)

def sign(x):
    return np.where(x >= 0, 1, -1)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(-z)**2

def generate_statistics(X, y):
    ks = [2, 5, 10]
    h = [3, 5, 10, 20]

    start = time.time()
    with open("estatisticas.txt", 'w') as f:
        for i in range(len(ks)):
            for j in range(len(h)):
                layers = (2, h[j], 1)
                merrs, mscores = (0, 0)
                for train, test in generate_kfolds(len(X), ks[i]):
                    nn = Network(layers)
                    nn.fit(X[train], y[train])
                    score = nn.score(X[test], y[test])
                    mscores += score
                    merrs += 1 - score
                mscores /= ks[i]
                merrs /= ks[i]
                f.write("%d %d %.4f %.4f\n" %(ks[i], h[j], merrs, mscores))
    end = time.time()
    print(end - start)
