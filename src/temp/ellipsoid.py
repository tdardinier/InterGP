import matplotlib.pyplot as plt
import numpy.random as rnd
from matplotlib.patches import Ellipse

class Ellipsoid:

    def __init__(self, x0, a, y0, b):

        # (x - x0)^2 / a^2 + (y - y0)^2 / b^2 <= 1

        self.x0 = x0
        self.y0 = y0
        self.a = a
        self.b = b

        self.ellipse = Ellipse(xy=[x0, y0], width=a, height=b, angle=0)
        self.ellipse.set_alpha(0.5)
        self.ellipse.set_facecolor(rnd.rand(3))


    def getX(self):
        a = self.x0 - 0.5 * self.a
        b = self.x0 + 0.5 * self.a
        return a, b

    def getY(self):
        a = self.y0 - 0.5 * self.b
        b = self.y0 + 0.5 * self.b
        return a, b


def combine(e1, e2):

    xa1, xb1 = e1.getX()
    ya1, yb1 = e1.getY()
    xa2, xb2 = e2.getX()
    ya2, yb2 = e2.getY()

    xa = min(xa1, xa2)
    xb = max(xb1, xb2)
    ya = min(ya1, ya2)
    yb = max(yb1, yb2)

    x0 = 0.5 * (xa + xb)
    y0 = 0.5 * (ya + yb)

    a = xb - xa
    b = yb - ya

    return Ellipsoid(x0, a, y0, b)


e1 = Ellipsoid(0.3, 0.3, 0.3, 0.4)
e2 = Ellipsoid(0.5, 0.4, 0.5, 0.3)
c = combine(e1, e2)

fig = plt.figure(0)
ax = fig.add_subplot(111)
ax.add_artist(e1.ellipse)
ax.add_artist(e2.ellipse)
ax.add_artist(c.ellipse)

plt.show()
