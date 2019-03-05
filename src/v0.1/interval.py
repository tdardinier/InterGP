class Interval:

    def __init__(self, a, b=None):
        self.a = a
        if b is None:
            self.b = a
        else:
            assert (a <= b)
            self.b = b

    def contains(self, x):
        return (self.a <= x) and (x <= self.b)

    @staticmethod
    def fromNoise(mean, noise):
        return Interval(mean - noise, mean + noise)

    @staticmethod
    def add(inter1, inter2):
        return Interval(inter1.a + inter2.a, inter1.b + inter2.b)

    @staticmethod
    def neg(inter):
        return Interval(-inter.b, -inter.a)

    @staticmethod
    def mult(inter1, inter2):
        a = inter1.a * inter2.a
        b = inter1.a * inter2.b
        c = inter1.b * inter2.a
        d = inter1.b * inter2.b
        m = min(a, b, c, d)
        M = max(a, b, c, d)
        return Interval(m, M)

    @staticmethod
    def sum(inters):
        s = Interval(0)
        for inter in inters:
            s = Interval.add(s, inter)
        return s

    def __str__(self):
        return "|" + str(self.a) + ", " + str(self.b) + "|"

    def __repr__(self):
        return self.__str__()
