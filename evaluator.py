import replayBuffer as rb
import numpy as np
import time


class Evaluator():

    def __init__(self, filename):
        self.buf = rb.ReplayBuffer(filename=filename)

    def crossValidate(self, classPredictor, k=10, c=1000):

        buf = self.buf.cut(c)

        n = len(buf.x[0])
        m = len(buf.u[0])
        results = []

        time_training = []
        for i in range(1, k):
            predictor = classPredictor.Predictor(n=n, m=m)
            (train, test) = buf.crossValidation(k, i)
            predictor.addData(train.x, train.u, train.y)
            t0 = time.time()
            print("Iteration", i, ", training...")
            predictor.train()
            print("Iteration", i, ", trained!", time.time() - t0, "s")
            time_training.append(time.time() - t0)
            c = 0
            num = len(test.x)

            t0 = time.time()
            for j in range(num):
                x = test.x[j]
                u = test.u[j]
                y = test.y[j]
                yy = predictor.predict(x, u)
                c += self.__norm(y, yy)

            print("Cost", c)

            results.append(c)
        print("Average time:", np.average(time_training))

        return results
