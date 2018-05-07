import tensorflow as tf
import numpy as np
import random as rd

nrd = np.random


def generateData():

    n = 2

    def generateX():
        return np.matrix([[rd.random() * 10] for _ in range(n)])

    def generateU():
        return rd.random() * 2 - 1

    def generateY(x, u):
        A = np.matrix([[0.2, 0.4],
                    [0.7, 0.1]])
        b = np.zeros([n, 1])
        b.put(0, -1.)
        eps = nrd.rand(n, 1)
        return A * x + u * b

    N = 500

    train_X = [generateX() for _ in range(N)]
    train_U = [generateU() for _ in range(N)]
    train_Y = [generateY(x, u) for (x, u) in zip(train_X, train_U)]

    return (train_X, train_U, train_Y)

class Predictor():

    def randomVar(self, n = 1, m = 1, name = "Undefined"):
        init = tf.random_uniform(shape=[n, m])
        return tf.Variable(init)

    def createNetwork(self):

        # self.batch_size = tf.placeholder("int", shape=(1), name="batch_size")

        self.X = tf.placeholder("float", shape=(None, self.n, 1), name="X")
        self.U = tf.placeholder("float", shape=(None, self.m, 1), name="U")
        self.Y = tf.placeholder("float", shape=(None, self.n, 1), name="Y")

        # self.rX = tf.reshape(self.X, [self.n, -1])
        self.rX = tf.reshape(tf.transpose(self.X, perm=[1, 2, 0]), [self.n, -1])
        # self.rU = tf.reshape(self.U, [-1, self.m, 1])
        # self.rY = tf.reshape(self.Y, [self.n, -1])
        self.rY = tf.reshape(tf.transpose(self.Y, perm=[1, 2, 0]), [self.n, -1])

        self.Wha = self.randomVar(self.k, self.n, name="Wha")
        self.Bha = self.randomVar(self.k, 1, name="Bha")
        self.Ha = tf.nn.relu(self.Wha @ self.rX + self.Bha, name="Ha")
        # self.Ha = tf.nn.relu(self.Wha @ self.rX, name="Ha")

        self.Wa = self.randomVar(self.n, self.k, name="Wa")
        self.Ba = self.randomVar(self.n, 1, name="Ba")
        self.A = self.Wa @ self.Ha + self.Ba
        # self.A = self.Wa @ self.Ha

        self.Whw = self.randomVar(self.k, self.n, name="Whw")
        self.Bhw = self.randomVar(self.k, 1, name="Bhw")
        self.Hw = tf.nn.relu(self.Wha @ self.rX + self.Bha, name="Hw")
        # self.Hw = tf.nn.relu(self.Wha @ self.rX, name="Hw")

        p = self.n * self.m
        self.Ww = self.randomVar(p, self.k, name="Wa")
        self.Bw = self.randomVar(p, 1, name="Ba")
        self.W = self.Ww @ self.Hw + self.Bw
        # self.W = self.Ww @ self.Hw

        self.rW = tf.reshape(tf.transpose(self.W), [-1, self.n, self.m])
        self.prod = self.rW @ self.U
        self.rProd = tf.reshape(tf.transpose(self.prod, [1, 2, 0]), [self.n, -1], name="rProd")

        # self.Pw = self.randomVar(self.n, self.k, name="Pw")
        # self.Qw = self.randomVar(1, self.m, name="Qw")
        # self.Bw = self.randomVar(self.n, self.m, name="Bw")
        # self.W = self.Pw @ self.Hw @ self.Qw + self.Bw

        # self.output = self.A + self.rW @ self.U
        self.output = self.A + self.rProd

        # Mean squared error
        self.cost = tf.reduce_mean(tf.pow(self.output - self.rY, 2), name="Cost")

        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        self.sess = tf.Session()

        self.sess.run(init)

    def printer(self, x):
        A = self.sess.run(self.A, feed_dict={self.X: [x]})
        W = self.sess.run(self.W, feed_dict={self.X: [x]})
        print("A:", A.shape)
        print("W:", W.shape)
        print("(n, m, k) =", (self.n, self.m, self.k))
        (n, m, k) = (self.n, self.m, self.k)
        print("Number of parameters:", 4 * n * k + 2 * k + n + m + n * m)

    def __init__(self, n = 2, k = 10, m = 1, online = True):
        self.n = n
        self.k = k
        self.m = m
        self.online = online
        self.data_X = []
        self.data_U = []
        self.data_Y = []
        self.createNetwork()

    def addData(self, x, u, y):
        self.data_X += x
        self.data_U += u
        self.data_Y += y

    def addElement(self, x, u, y):
        self.data_X.append(x)
        self.data_U.append(u)
        self.data_Y.append(y)

    def getParameters(self, x):
        A = self.sess.run(self.A, feed_dict={self.X:x})
        B = self.sess.run(self.By, feed_dict={self.X:x})
        W = self.sess.run(self.Wy, feed_dict={self.X:x})
        return (A, B, W)

    def train_online(self, x, u, y):
        _, c = self.sess.run([self.optimizer, self.cost],  feed_dict={self.X: x,  self.U: u, self.Y: y})
        return c

    def train(self, n_epochs = 100, n_max = 1000):

        BATCH_SIZE = 2
        n = len(self.data_X)
        indices = [i for i in range(n)]
        n = min(n, (n_max // BATCH_SIZE) * BATCH_SIZE)
        n_batchs = n // BATCH_SIZE
        print("n", n, ", n_batchs", n_batchs)

        for epoch in range(n_epochs):

            rd.shuffle(indices)
            cost = 0

            for batch in range(n_batchs):

                a = batch * BATCH_SIZE
                b = (batch + 1) * BATCH_SIZE
                r = range(a, b)
                # r = [0, 1, a]
                # self.data_X[0] = [[0], [0], [0], [0]]
                # self.data_U[0] = [[0]]
                # self.data_Y[0] = [[0], [0], [0], [0]]
                # big = 1000
                # self.data_X[1] = [[big], [big], [big], [big]]
                # self.data_U[1] = [[big]]
                # self.data_Y[1] = [[big], [big], [big], [big]]
                x = np.asarray([self.data_X[i] for i in r])
                u = np.asarray([self.data_U[i] for i in r])
                y = np.asarray([self.data_Y[i] for i in r])

                d = {self.X: x, self.U: u, self.Y: y}

                # print("")
                # print("")
                # print("SAPART")
                # print("x", x)
                # print("rX", self.sess.run(self.rX, feed_dict=d))
                # print("A", self.sess.run(self.A, feed_dict=d))
                # print("rProd", self.sess.run(self.rProd, feed_dict=d))
                # print("rW", self.sess.run(self.rW, feed_dict=d))
                # print("W", self.sess.run(self.W, feed_dict=d))
                # print("U", self.sess.run(self.U, feed_dict=d))

                out = self.sess.run(self.output, feed_dict=d)
                rX = self.sess.run(self.rX, feed_dict=d)
                rY = self.sess.run(self.rY, feed_dict=d)
                # print("rX", rX)
                # print("u", u)
                # print("rY", rY)
                # print("mult", self.sess.run(self.rW @ self.U, feed_dict=d))
                # print("out", out)

                _, c = self.sess.run([self.optimizer, self.cost], feed_dict=d)
                cost += c

            print("EPOCH:", epoch, ", COST:", cost)

    def evaluate(self, x, u):
        return self.sess.run(self.output, feed_dict={self.X: [x], self.U: [u]})[0]


# p = Predictor(2)
# (x, u, y) = generateData()
# p.addData(x, u, y)
