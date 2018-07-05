import numpy as np


class Trajectory:

    def __init__(self):

        self.X = []
        self.S = []
        self.aimedP = []
        self.realP = []
        self.U = []

    def addStart(self, x_0):
        self.X.append(x_0)
        self.S.append([(xx, xx) for xx in x_0])

    def addBuf(self, buf, k=None):
        if k is None:
            k = len(self.S) - 1
        self.X = [buf.x[i] for i in range(k + 1)]

    def addU(self, U):
        self.U = U

    def addPrediction(self, SS, aimedP, realP):
        self.S.append(SS)
        self.aimedP.append(aimedP)
        self.realP.append(realP)

    def __log(self, s):
        print("Trajectory: " + s)

    def save(self, filename):
        a = np.array([self.X, self.S, self.aimedP, self.realP, self.U])
        self.__log("Saving " + filename + "...")
        np.save(filename, a)
        self.__log("Saved!")

    def load(self, filename):
        self.__log("Loading " + filename + "...")
        a = np.load(filename)
        self.__log("Loaded!")

        self.X = list(a[0])
        self.S = list(a[1])
        self.aimedP = list(a[2])
        self.realP = list(a[3])
        self.U = list(a[4])
