import replayBuffer as rb
import result as rs
import tools


class Evaluator():

    def __init__(self, classPredictor, env_name, agent_name):

        self.classPredictor = classPredictor
        self.env_name = env_name
        self.agent_name = agent_name

        replay_filename = tools.FileNaming.replayName(env_name, agent_name)
        self.buf = rb.ReplayBuffer(filename=replay_filename)

    def sampleValidate(self, c=10000, k=100):

        n_test = int(c / k)

        buf = self.buf.shuffle()
        buf = buf.cut(c + n_test)
        buf = buf.normalize()

        n = len(buf.x[0])
        m = len(buf.u[0])
        r = rs.Result(n=n, m=m)

        train = buf.slice([(0, c)])
        test = buf.slice([(c, c + n_test)])

        r.beginTimer()

        predictor = self.classPredictor.Predictor(n=n, m=m)
        predictor.addData(train.x, train.u, train.y)
        predictor.train()

        i = 0
        for (x, u, y) in zip(test.x, test.u, test.y):
            sigma = None
            i += 1
            if predictor.std:
                (yy, sigma) = predictor.predict(x, u)
            else:
                yy = predictor.predict(x, u)
            print(i)
            r.addResults(x, u, y, yy, sigma)

        f = tools.FileNaming.resultName(
            predictor.name, self.env_name, self.agent_name, c)
        r.saveTimer()
        r.save(f)
        return r

    def crossValidate(self, c=1000, k=10):

        buf = self.buf.cut(c)
        buf = buf.shuffle()
        buf = buf.normalize()
        n = len(buf.x[0])
        m = len(buf.u[0])
        r = rs.Result(k=k, c=c, n=n, m=m)

        predictor = None
        # for i in range(k):
        for i in range(0, 1):
            predictor = self.classPredictor.Predictor(n=n, m=m)
            (train, test) = buf.crossValidation(k, i)
            predictor.addData(train.x, train.u, train.y)
            print("Evaluator: Iteration " + str(i + 1) + "/" + str(k))
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

        f = tools.FileNaming.resultName(
            predictor.name, self.env_name, self.agent_name, c)
        r.save(f)
        return r
