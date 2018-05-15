class Predictor():

    def __init__(self, n, m):

        self.n = n
        self.m = m
        self.std = False

        self.data_X = []
        self.data_U = []
        self.data_Y = []

        self.name = "predictor"

        self.clear()

    def clear(self):
        self.data_X = []
        self.data_U = []
        self.data_Y = []

    def addData(self, X, U, Y):
        self.data_X += X
        self.data_U += U
        self.data_Y += Y

    def addElement(self, x, u, y):
        self.data_X.append(x)
        self.data_U.append(u)
        self.data_Y.append(y)

    def train(self):
        print("Not implemented yet")

    def predict(self, x, u):
        print("Not implemented yet")
