import tensorflow as tf
import numpy as np
import random as rd

class Predictor():

    def createNetwork(self):

        n = 2

        self.X = tf.placeholder("float", shape=(n, 1))
        self.U = tf.placeholder("float", shape=())
        self.Y = tf.placeholder("float", shape=(n, 1))

        self.A2 = tf.Variable([[np.random.randn()] for _ in range(n)])
        self.A1 = tf.Variable([[np.random.randn() for _ in range(n)] for _ in range(n)])
        self.A0 = tf.Variable([[np.random.randn()] for _ in range(n)])

        self.B2 = tf.Variable([[np.random.randn()] for _ in range(n)])
        self.B1 = tf.Variable([[np.random.randn() for _ in range(n)] for _ in range(n)])
        self.B0 = tf.Variable([[np.random.randn()] for _ in range(n)])

        Xt = tf.transpose(self.X)
        X2 = self.X @ Xt
        self.A = X2 @ self.A2 + self.A1 @ self.X + self.A0
        self.B = X2 @ self.B2 + self.B1 @ self.X + self.B0
        self.pred = self.A + tf.multiply(self.U, self.B)

        # Mean squared error
        cost = tf.reduce_mean(tf.pow(self.pred - self.Y, 2)) / 2

        self.optimizer = tf.train.AdamOptimizer().minimize(cost)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        self.sess = tf.Session()

        self.sess.run(init)

    def __init__(self, n = 2):
        self.n = n
        self.createNetwork()
        self.generateData()
        self.train()

    def generateData(self):

        def generateX():
            return np.matrix([[rd.random() * 10] for _ in range(self.n)])

        def generateU():
            return rd.random() * 2 - 1

        def generateY(x, u):
            A = np.matrix([[0.2, 0.4],
                        [0.7, 0.1]])
            b = np.zeros([self.n, 1])
            b.put(0, -1.)
            eps = np.random.rand(self.n, 1)
            return A * x + u * b

        N = 500

        self.train_X = np.asarray([generateX() for _ in range(N)])
        self.train_U = np.asarray([generateU() for _ in range(N)])
        self.train_Y = np.asarray([generateY(x, u) for (x, u) in zip(self.train_X, self.train_U)])

    def getParameters(self, x):
        A = self.sess.run(self.A, feed_dict={self.X:x})
        B = self.sess.run(self.B, feed_dict={self.X:x})
        return (A, B)

    def train(self, training_epochs = 100):

        for epoch in range(training_epochs):

            for (x, u, y) in zip(self.train_X, self.train_U, self.train_Y):
                self.sess.run(self.optimizer,  feed_dict={self.X: x,  self.U: u, self.Y: y})

            if (epoch+1) % 10 == 0:
               print("A1:", self.sess.run(self.A1))
               print("B0:", self.sess.run(self.B0))
