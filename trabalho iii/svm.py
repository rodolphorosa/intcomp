import numpy as np
from sklearn import svm

from common import *
from plot import *

def generate_statistics(X, y):
    statistics = np.zeros((3, 7))
    ks = [2, 5, 10]
    kernels = [
        ("sigmoid", 1),
        ("sigmoid", 0.5),
        ("sigmoid", 0.01),
        ("linear"),
        ("poly"),
        ("rbf")]

    statistics[:, 0] = np.array(ks)

    for i in range(len(ks)):
        for train, test in generate_kfolds(len(X), ks[i]):
            for j in range(len(kernels)):
                try:
                    krn, g = kernels[j]
                    clf = svm.SVC(kernel=krn, gamma=g)
                except:
                    krn = kernels[j]
                    clf = svm.SVC(kernel=krn)
                finally:
                    clf.fit(X[train], y[train])
                    statistics[i, j + 1] += (1 - clf.score(X[test], y[test]))
        statistics[i, 1:] /= ks[i]

    header = "K,Sigmoid1,Sigmoid2,Sigmoid3,Linear,Polynomial,RBF"
    np.savetxt("statistics.txt",
        statistics, fmt="%.4f", delimiter=",", header=header, comments='')