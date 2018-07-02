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

    def getListe(self):

        liste = []

        def strBool(b, s=""):
            if b:
                return s + "1"
            return s + "0"

        liste.append(strBool(self.scipy, "sci"))
        liste.append(strBool(self.centered, "cen"))
        liste.append(strBool(self.matern, "mat"))
        liste.append(strBool(self.noise, "noi"))
        liste.append(strBool(self.riskAllocUniform, "unif"))
        liste.append(strBool(self.probTransition, "trans"))

        return liste
