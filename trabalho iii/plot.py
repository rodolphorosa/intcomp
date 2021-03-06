import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from common import *

def plot_dispersal(X, y):
    px = X[y > 0, 0]
    py = X[y > 0, 1]

    nx = X[y < 0, 0]
    ny = X[y < 0, 1]

    pos = plt.scatter(px, py, c="b", s=20, edgecolors="none")
    neg = plt.scatter(nx, ny, c="r", s=20, edgecolors="none")

    plt.axis('tight')
    plt.legend((pos, neg), ("+1", "-1"), loc="upper left")
    plt.show()

def plot_margin(X, y, clf, title):
    plt.figure("SVM")
    plt.clf()

    px = X[y > 0, 0]
    py = X[y > 0, 1]
    nx = X[y < 0, 0]
    ny = X[y < 0, 1]

    clf.fit(X, y)
    sv = clf.support_vectors_

    plt.scatter(nx, ny, c="r", s=20, edgecolors="none")
    plt.scatter(px, py, c="b", s=20, edgecolors="none")
    plt.scatter(sv[:, 0], sv[:, 1], s=80, facecolors="none", edgecolors='k')

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)

    cs = plt.contour(XX, YY, Z,
        colors=['k', 'k', 'k'],
        linestyles=['--', '-', '--'],
        levels=[-1, 0, 1])
    # cs = plt.contour(XX, YY, Z, colors='k')
    
    plt.legend(("+1", "-1", "Vetores de suporte"), loc="upper left")
    plt.show()

def plot_decision(X, y, clf):
    plt.figure("SVM")
    plt.clf()

    px = X[y > 0, 0]
    py = X[y > 0, 1]
    nx = X[y < 0, 0]
    ny = X[y < 0, 1]

    plt.scatter(nx, ny, c="r", zorder=10, s=20, edgecolors="none")
    plt.scatter(px, py, c="b", zorder=10, s=20, edgecolors="none")

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)

    cs = plt.contour(XX, YY, Z)
    plt.legend(("+1", "-1"))
    plt.show()