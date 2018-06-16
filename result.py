import numpy as np
import time


class Result():

    def __init__(self, k=10, c=1000, n=4, m=1, filename=None):
        self.k = k
        self.c = c
        self.n = n
        self.m = m
        self.x = np.empty([0, self.n])
        self.u = np.empty([0, self.m])
        self.real_y = np.empty([0, self.n])
        self.predicted_y = np.empty([0, self.n])
        self.time = np.array([])
        self.sigma = np.empty([0, self.n])
        # self.sigma = np.empty([0, 1]) UNSEPARATED
        self.t0 = None

        if filename is not None:
            self.load(filename=filename)

    def getLengthFirstTrajectory(self):
        prev_y = self.x[0]
        found = False
        i = 0
        while not found:
            x = self.x[i]
            y = self.real_y[i]
            if np.array_equal(x, prev_y):
                prev_y = y
            else:
                found = True
            i += 1
        return i

    def addResults(self, x, u, real_y, predicted_y, sigma=None):
        self.x = np.vstack([self.x, np.array(x).T])
        self.u = np.vstack([self.u, np.array(u).T])
        self.real_y = np.vstack([self.real_y, np.array(real_y).T])
        self.predicted_y = np.vstack([
            self.predicted_y,
            np.array(predicted_y).T])
        if sigma is not None:
            self.sigma = np.vstack([self.sigma, sigma])

    def beginTimer(self):
        self.t0 = time.time()
        print("Result: Launching timer...")

    def saveTimer(self):
        t = time.time() - self.t0
        print("Result: Saving timer: " + str(t))
        self.time = np.append(self.time, t)

    def addX(self):
        ry = [self.real_y[i] + self.x[i] for i in range(len(self.x))]
        py = [self.predicted_y[i] + self.x[i] for i in range(len(self.x))]
        self.real_y = ry
        self.predicted_y = py

    def save(self, filename):
        print("Result: Saving " + filename + "...")
        k = np.array(self.k)
        c = np.array(self.c)
        n = np.array(self.n)
        m = np.array(self.m)
        np.savez(filename, k=k, c=c, n=n, m=m,
                 x=self.x, u=self.u,
                 real_y=self.real_y, predicted_y=self.predicted_y,
                 time=self.time, sigma=self.sigma)
        print("Result: Saved!")

    def load(self, filename):
        print("Result: Loading " + filename + "...")
        f = np.load(filename)
        self.k = int(f['k'])
        self.c = int(f['c'])
        self.n = int(f['n'])
        self.m = int(f['m'])
        self.x = f['x']
        self.u = f['u']
        self.real_y = f['real_y']
        self.predicted_y = f['predicted_y']
        self.time = f['time']
        self.sigma = f['sigma']
        print("Result: Loaded!")
