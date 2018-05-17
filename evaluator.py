import replayBuffer as rb
import result as rs
import tools


class Evaluator:

    def __init__(self, predictor, env_name, agent_name):

        self.predictor = predictor
        self.env_name = env_name
        self.agent_name = agent_name

        replay_filename = tools.FileNaming.replayName(env_name, agent_name)
        self.buf = rb.ReplayBuffer(filename=replay_filename)

    def sampleValidate(self, c=10000, n_test=1000):

        buf = self.buf.shuffle()
        buf = buf.cut(c + n_test)
        buf = buf.normalize()
        buf = buf.removeX()  # Removing X

        n = len(buf.x[0])
        m = len(buf.u[0])
        r = rs.Result(n=n, m=m)

        train = buf.slice([(0, c)])
        test = buf.slice([(c, c + n_test)])

        r.beginTimer()

        predictor = self.predictor(n=n, m=m)
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
                if i % 100 == 0:
                    print("Evaluator: " + str(i) + "/" + str(n_test))
            r.addResults(x, u, y, yy, sigma)

        f = tools.FileNaming.resultName(
            predictor.name, self.env_name, self.agent_name, c)
        r.saveTimer()
        r.addX()  # Adding X
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
        for i in range(k):
            predictor = self.predictor(n=n, m=m)
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
