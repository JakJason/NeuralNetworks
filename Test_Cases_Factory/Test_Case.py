import random as rand
import numpy as np
import matplotlib.pyplot


class TestCase:
    def __init__(self):

        self.period = 1
        self.n_points = 1
        self.points_x = []
        self.points_y = []
        self.factors = [0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0]

    def function(self, x):
        y = self.factors[0]/2
        for i in range(1, len(self.factors)):
            y = y + (self.factors[i] * np.sin(x * ((2 * i * np.pi)/self.period)))
        return y

    def set_random(self, n_points):
        self.points_x = []
        self.points_y = []
        self.period = rand.uniform(10, 50)
        self.n_points = n_points
        for i in range(0, len(self.factors)):
            self.factors[i] = rand.uniform(-20, 20)
        for i in range(0, n_points):
            x = rand.uniform(0, self.period)
            self.points_x.append(x)

        self.points_x.sort()

        for i in range(0, n_points):

            y = self.function(self.points_x[i])
            self.points_y.append(y)

    def set_from_file(self, file_path):
        f = open(file_path, "r")
        data = f.readlines()
        for i in range(0, len(data)):
            head, sep, tail = data[i].partition(' ')
            data[i] = head
        self.period = float(data[0])
        self.n_points = int(data[2])
        for i in range(0, len(self.factors)):
            self.factors[i] = float(data[i + 4])
        for i in range(0, self.n_points):
            self.points_x.append(float(data[30 + i*2]))
            self.points_y.append(float(data[31 + i*2]))

    def save_to_file(self, file_path):
        f = open(file_path, "w+")
        f.write("%f \n\n" % self.period)
        f.write("%d \n\n" % self.n_points)
        for i in range(0, len(self.factors)):
            f.write("%f \n" % self.factors[i])
        f.write("\n")
        for i in range(0,self.n_points):
            f.write("%f \n%f \n" % (self.points_x[i], self.points_y[i]))

    def save_to_csv(self, file_path):
        f = open(file_path, "a")

        f.write("%f," % self.period)
        for i in range(0, len(self.factors)):
            f.write("%f," % self.factors[i])
        for i in range(0,self.n_points):
            f.write("%f,%f," % (self.points_x[i], self.points_y[i]))
        f.write("\n")

    def show(self):
        x = np.linspace(0, self.period, 201)

        matplotlib.pyplot.plot(x, self.function(x), 'b', self.points_x, self.points_y, 'r^')
        matplotlib.pyplot.show()


if __name__ == "__main__":
    case = TestCase()
    for i in range(100000):
        print(i)
        case.set_random(100)
        case.save_to_csv("dataset4.csv")
