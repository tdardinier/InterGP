import replayBuffer as rb
import result as rs


class Evaluator():

    def __init__(self, filename):
        self.buf = rb.ReplayBuffer(filename=filename)

    def crossValidate(self, classPredictor, k=10,
                      c=1000, filename="undefined"):

        buf = self.buf.cut(c)
        n = len(buf.x[0])
        m = len(buf.u[0])
        r = rs.Result(k=k, c=c, n=n, m=m)

        for i in range(k):
            predictor = classPredictor.Predictor(n=n, m=m)
            (train, test) = buf.crossValidation(k, i)
            predictor.addData(train.x, train.u, train.y)
            print("Iteration", i)
            r.beginTimer()
            predictor.train()
            r.saveTimer()
            num = len(test.x)

            for j in range(num):
                x = test.x[j]
                u = test.u[j]
                y = test.y[j]
                sigma = None
                if predictor.std:
                    (yy, sigma) = predictor.predict(x, u)

                else:
                    yy = predictor.predict(x, u)
                r.addResults(x, u, y, yy, sigma)

        r.save(filename)
        return r
