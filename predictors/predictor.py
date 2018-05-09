import tensorflow as tf
import numpy as np
import random as rd

nrd = np.random


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

        self.Wha1 = self.randomVar(self.k, self.n, name="Wha1")
        self.Bha1 = self.randomVar(self.k, 1, name="Bha1")
        self.Ha1 = tf.nn.relu(self.Wha1 @ self.rX + self.Bha1, name="Ha1")

        self.Wha2 = self.randomVar(self.k, self.k, name="Wha2")
        self.Bha2 = self.randomVar(self.k, 1, name="Bha2")
        self.Ha2 = tf.nn.relu(self.Wha2 @ self.Ha1 + self.Bha2, name="Ha2")

        self.Wa = self.randomVar(self.n, self.k, name="Wa")
        self.Ba = self.randomVar(self.n, 1, name="Ba")
        self.A = self.Wa @ self.Ha2 + self.Ba

        self.Whw1 = self.randomVar(self.k, self.n, name="Whw1")
        self.Bhw1 = self.randomVar(self.k, 1, name="Bhw1")
        self.Hw1 = tf.nn.relu(self.Whw1 @ self.rX + self.Bhw1, name="Hw1")

        self.Whw2 = self.randomVar(self.k, self.k, name="Whw2")
        self.Bhw2 = self.randomVar(self.k, 1, name="Bhw2")
        self.Hw2 = tf.nn.relu(self.Whw2 @ self.Hw1 + self.Bhw2, name="Hw2")

        p = self.n * self.m
        self.Ww = self.randomVar(p, self.k, name="Wa")
        self.Bw = self.randomVar(p, 1, name="Ba")
        self.W = self.Ww @ self.Hw2 + self.Bw

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
        # self.cost = tf.reduce_mean(tf.pow(self.output - self.rY, 2), name="Cost")
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

    def __init__(self, n = 2, k = 30, m = 1, online = True):
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

    def train(self, n_epochs = 500, n_max = 10000):

        BATCH_SIZE = 32
        epsilon = 0.00001
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
                x = np.asarray([self.data_X[i] for i in r])
                u = np.asarray([self.data_U[i] for i in r])
                y = np.asarray([self.data_Y[i] for i in r])

                d = {self.X: x, self.U: u, self.Y: y}

                _, c = self.sess.run([self.optimizer, self.cost], feed_dict=d)
                cost += c / n_batchs

                # out = self.sess.run(self.output, feed_dict=d)
                # rX = self.sess.run(self.rX, feed_dict=d)
                # rY = self.sess.run(self.rY, feed_dict=d)
                # print("rX", rX)
                # print("out", out)
                # print("rY", rY)

            print("EPOCH:", epoch, ", COST:", cost, "BATCH_SIZE:", BATCH_SIZE)
            if cost < epsilon:
                break

    def evaluate(self, x, u, y):
        d = {self.X: [x], self.U: [u], self.Y: [y]}
        return self.sess.run(self.output, feed_dict=d)


# p = Predictor(2)
# (x, u, y) = generateData()
# p.addData(x, u, y)
