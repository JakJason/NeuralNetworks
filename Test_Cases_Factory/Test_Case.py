import random as rand
import numpy as np
import matplotlib.pyplot


class TestCase:
    def __init__(self):
        self.period = 1
        self.points_x = []
        self.points_y = []
        self.factors = [0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0]

    def function(self, x):
        y = self.factors[0]/2
        for i in range(1, 25):
            y = y + (self.factors[i] * np.sin(x * ((2 * i * np.pi)/self.period)))
        return y

    def set_random(self):
        self.period = rand.uniform(10, 50)
        for i in range(0, 25):
            self.factors[i] = rand.uniform(-20, 20)
        for i in range(0, 100):
            x = rand.uniform(0, self.period)
            y = self.function(x)
            self.points_x.append(x)
            self.points_y.append(y)

    def show(self):
        x = np.linspace(0, self.period, 201)
        matplotlib.pyplot.plot(x, self.function(x), 'b', self.points_x, self.points_y, 'r^')
        matplotlib.pyplot.show()


if __name__ == "__main__":
    case = TestCase()
    case.set_random()
    case.show()
