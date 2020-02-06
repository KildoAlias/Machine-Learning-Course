import numpy as np
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# TODO : Call minimize
# TODO : Extract the non-zero α values
# TODO : Calculate the b value using equation (7).


class SVM():

    def __init__(self, start, upperBound, samples, target, data):
        self.start = start
        self.bounds = [(0, upperBound) for b in range(samples)]
        self.constraints = {'type': 'eq', 'fun': self.zeroFunction}
        self.data = data
        self.target = target
        self.samples = samples
        self.P = self.Pmatrix()
        self.sv = []
        self.b = None
        self.bBit = []

    # def __str__(self):
    #     return "alpha: " + str(self.alpha)

    #  A linear kernel function

    def Pmatrix(self):
        self.P = np.empty([self.samples, self.samples])
        for i in range(self.samples):
            for j in range(self.samples):
                self.P[i, j] = self.target[i]*self.target[j] * \
                    self.linearKernalFunction(self.data[i], self.data[j])

        return self.P

    def linearKernalFunction(self, x, y):
        K = np.dot(x, y)
        return K

    #  A polynomial kernel function

    def polyKernalFunction(self, x, y, p):
        K = np.power((np.dot(x, y)+1), p)
        return K

    # A Radial Basis Function (RBF) kernels
    def RBFKernalFunction(self, x, y, sigma):
        K = np.exp(-(np.pow((np.linalg.norm(x-y)), 2))/(2*sigma**2))
        return K

    # The function objective

    def objectiveFunction(self, alpha):
        objective = 0.5*(np.sum([
            np.dot(np.dot(np.transpose([alpha]), [alpha]), self.P)]))-np.sum(alpha)
        return objective

    #  zeroFunction constraint

    def zeroFunction(self, alpha):
        zero = np.dot(self.target, alpha)
        return zero

    # The indicator function

    def indicator(self, x, y):
        self.threshold(x, y)
        ind = np.subtract(self.bBit, self.b)
        return ind

    # Threshold
    def threshold(self, x, y):
        bv = []
        for j in range(len(self.sv[0])):
            self.bBit.append(np.dot(self.sv[2][j], np.dot(
                self.sv[1][j], self.linearKernalFunction(self.sv[0][j], [x, y]))))
            bv.append(self.bBit-self.sv[1][j])
        self.b = np.sum(bv)
        return self.b

    def mini(self):
        ret = minimize(self.objectiveFunction, self.start,
                       bounds=self.bounds, constraints=self.constraints)
        self.alpha = ret['x']

        # self.alpha = np.array([a*int(a<10**-8) for a in self.alpha if a < 10**-8])
        for i, e in enumerate(self.alpha):
            if e < 10**-8:
                self.alpha[i] = 0

        idx = np.nonzero(self.alpha)[0]
        x = []
        t = []
        a = []
        for i in idx:
            x.append(self.data[i])
            t.append(self.target[i])
            a.append(self.alpha[i])

        self.sv = (x, t, a)
        return self.sv

    # TODO : Code for generating test data and for visualizing the results.
    # TODO : Generating Test Data
    # TODO : Plotting


def generateTestData():
    classA = np.concatenate((np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
                             np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
    classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate(
        (np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

    samples = inputs.shape[0]
    permute = list(range(samples))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]
    return classA, classB, inputs, targets, samples


def plot(classA, classB):
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')

    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
    plt.axis('equal')
    plt.savefig('svmplot.pdf')
    return


def DecisionBoundary(svm):
    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)

    grid = np.array([[svm.indicator(x, y) for x in xgrid]for y in ygrid])

    plt.contour(xgrid, ygrid, grid, (-1, 0, 1),
                colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))


def main():
    np.random.seed(100)
    classA, classB, inputs, targets, samples = generateTestData()
    start = np.zeros(samples)
    upperBound = None
    svm = SVM(start, upperBound, samples, targets, inputs)
    sv = svm.mini()
    plot(classA, classB)
    DecisionBoundary(svm)
    plt.show()


if __name__ == "__main__":
    main()
