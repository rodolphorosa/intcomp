from numpy import loadtxt, asmatrix
from numpy.random import permutation

PATH = "../data/banana.dat"

def generate_data():
    data = loadtxt(PATH, dtype="float", skiprows=7, delimiter=",")
    X = data[:, :2]
    y = data[:, 2]
    return (X, y)

def generate_kfolds(n, k=5):
    perm = permutation(n)
    size = n//k
    for i in range(k):
        test  = perm[i*size:(i+1)*size]
        train = [perm[j] for j in perm if perm[j] not in test]
        yield train, test

def divide_train_test(n, test=.2):
    cut = int(n * test)
    perm = permutation(n)
    return perm[cut:], perm[:cut]
