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
