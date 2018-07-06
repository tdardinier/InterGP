class Conf:

    def __init__(self,
                 k=None,                    # kernel function
                 n=1,                       # dimension of state
                 m=1,                       # dimension of action
                 debug=False,               # print logs?
                 scipy=True,                # use scipy or homemade GP
                 centered=False,            # prediction is not centered
                 matern=False,              # if not matern => SE
                 noise=False,               # add noise to the kernel if scipy
                 riskAllocUniform=False,    # how to allocate risk
                 probTransition=True,       # prob given is total/transition
                 seed=42,                   # random seed
                 epsilon=1e-20,             # added to make matrices singular
                 epsilon_f=0.000001,        # assertions for determinism
                 max_iter_minimizer=200,    # minimizer isn't perfect
                 test_chaos_theory=True,   # always constant variance
                 k_begin_chaos=2,           # when to begin
                 ):

        self.k = k
        self.n = n
        self.m = m
        self.debug = debug
        self.scipy = scipy
        self.centered = centered
        self.matern = matern
        self.noise = noise
        self.riskAllocUniform = riskAllocUniform
        self.probTransition = probTransition
        self.seed = seed
        self.epsilon = epsilon
        self.epsilon_f = epsilon_f
        self.max_iter_minimizer = max_iter_minimizer
        self.test_chaos_theory = test_chaos_theory
        self.k_begin_chaos = k_begin_chaos

        self.descr = []
        self.descr.append([self.scipy, "Scipy", "sci"])
        self.descr.append([self.centered, "Centered", "cen"])
        self.descr.append([self.matern, "Matern", "mat"])
        self.descr.append([self.noise, "Noise", "noi"])
        self.descr.append([self.riskAllocUniform, "Risk uniform", "unif"])
        self.descr.append([self.probTransition, "Prob transition", "trans"])

    def getListe(self, i=2, v='1', f='0'):

        liste = []

        def strBool(b, s=""):
            if b:
                return s + v
            return s + f

        for t in self.descr:
            liste.append(strBool(t[0], t[i]))

        if self.test_chaos_theory:
            if i == 1:
                liste.append("Test chaos" + v)
            else:
                liste.append("testChaos")

        return liste

    def getDescrName(self):
        liste = self.getListe(1, ': 1', ': 0')
        return ' - '.join(liste)
