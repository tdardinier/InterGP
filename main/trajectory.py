import numpy as np


class Trajectory:

    def __init__(self, S, P, buf):

        self.S = S
        self.P = [1.] + P
        self.X = [buf.x[i] for i in range(len(S))]

    def __log(self, s):
        print("Trajectory: " + s)

    def save(self, filename):
        a = np.array([self.X, self.S, self.P])
        self.__log("Saving " + filename + "...")
        np.save(filename, a)
        self.__log("Saved!")

    def load(self, filename):
        self.__log("Loading " + filename + "...")
        a = np.load(filename)
        self.__log("Loaded!")

        self.X = list(a[0])
        self.S = list(a[1])
        self.P = list(a[2])
