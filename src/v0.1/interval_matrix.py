import numpy as np
from oldInterGP.interval import Interval


class InterMatrix:

    # M = array of intervals
    def __init__(self, M):
        self.n = len(M)
        self.m = len(M[0])
        self.matrix = M

    @staticmethod
    def zeros(n, m):
        M = [[Interval(0.0) for j in range(m)] for i in range(n)]
        return InterMatrix(M)

    def transpose(self):
        T = [[self.matrix[i][j] for i in range(self.n)] for j in range(self.m)]
        return InterMatrix(T)

    def neg(self):
        N = [[Interval.neg(self.matrix[i][j])
              for j in range(self.m)] for i in range(self.n)]
        return InterMatrix(N)

    @staticmethod
    def createFromMatrix(M):
        n = np.shape(M)[0]
        m = np.shape(M)[1]
        iM = [[Interval(M[i, j]) for j in range(m)] for i in range(n)]
        return InterMatrix(iM)

    @staticmethod
    def add(M1, M2):
        assert (M1.n == M2.n)
        assert (M1.m == M2.m)
        M = [[0 for _ in range(M1.m)] for _ in range(M1.n)]
        for i in range(M1.n):
            for j in range(M1.m):
                M[i][j] = Interval.add(M1.matrix[i][j], M2.matrix[i][j])
        return InterMatrix(M)

    @staticmethod
    def mult(M1, M2):
        assert (M1.m == M2.n)
        c = M1.m
        M = [[Interval(0) for _ in range(M2.m)] for _ in range(M1.n)]
        for i in range(M1.n):
            for j in range(M2.m):
                for k in range(c):
                    p = Interval.mult(M1.matrix[i][k], M2.matrix[k][j])
                    M[i][j] = Interval.add(M[i][j], p)
        return InterMatrix(M)

    def __str__(self):
        s = ""
        for i in range(self.n):
            s += " ["
            for j in range(self.m):
                s += self.matrix[i][j].__str__() + ", "
            s = s[:-2] + "],\n"
        return "[" + s[1:-2] + "]"

    def __repr__(self):
        return self.__str__()


n = 2
m = 2
Z = InterMatrix.zeros(n, m)
M = InterMatrix.createFromMatrix(np.matrix([[1, 2], [3, 4]]))
Id = InterMatrix.createFromMatrix(np.matrix([[1, 0], [0, 1]]))
