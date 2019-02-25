import random as rand
import numpy as np
import matplotlib.pyplot
import csv


class TestCase:

    def __init__(self):

        self.period = 1
        self.n_points = 100
        self.points_x = []
        self.points_y = []
        # self.factors = [0, 0, 0, 0, 0,
        #                 0, 0, 0, 0, 0,
        #                 0, 0, 0, 0, 0,
        #                 0, 0, 0, 0, 0,
        #                 0, 0, 0, 0, 0,
        #                 0, 0, 0, 0, 0,
        #                 0, 0, 0, 0, 0,
        #                 0, 0, 0, 0, 0]
        # self.factors = [0, 0, 0, 0, 0,
        #                 0, 0, 0, 0, 0,
        #                 0, 0, 0, 0, 0,
        #                 0, 0, 0, 0, 0]
        self.factors = [0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0]
        self.av_y = 0
        self.av_x = 0

    def function(self, x):
        y = self.av_y
        for i in range(0, int(len(self.factors)/2)):
            y = y + (self.factors[i*2] * np.sin(x * ((2 * (i+1) * np.pi)/self.period)))\
                    + (self.factors[(i*2)+1] * np.cos(x * ((2 * (i+1) * np.pi)/self.period)))

        return y

    def set_random(self, n_points, max_factor):
        self.points_x = []
        self.points_y = []
        self.period = rand.uniform(10, 50)
        self.n_points = n_points
        for i in range(0, len(self.factors)):
            self.factors[i] = rand.uniform(-max_factor, max_factor)
        for i in range(0, n_points):
            x = (i * self.period)/self.n_points
            self.points_x.append(x)
        self.points_x.sort()
        for i in range(0, n_points):
            y = self.function(self.points_x[i])
            self.points_y.append(y)

    def set_simple_sinus(self):
        self.period = rand.uniform(10, 50)
        self.factors = [0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0]
        self.factors[0] = 10
        for i in range(0, self.n_points):
            x = (i * self.period)/self.n_points
            self.points_x.append(x)
        self.points_x.sort()
        for i in range(0, self.n_points):
            y = self.function(self.points_x[i])
            self.points_y.append(y)

    def set_from_csv(self, file_path):
        with open(file_path) as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                self.period = row[0]
                self.factors = row[1:len(self.factors)+1]
                print(len(row[1:len(self.factors)+1]))
                print(self.n_points)
                print(len(row))
                for i in range(self.n_points):
                     self.points_x.append(row[len(self.factors)+1 + i*2])
                     self.points_y.append(row[len(self.factors)+2 + i*2])
                break

    def set_from_result(self, period, factors, av, points_x, points_y):
        self.period = period
        self.factors = factors
        self.points_x = points_x
        self.points_y = points_y
        self.av_y = av

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
        x = np.linspace(self.points_x[0], self.points_x[-1], 201)
        matplotlib.pyplot.plot(x, self.function(x), 'b', label="Funkcja S(x)")
        matplotlib.pyplot.plot(self.points_x, self.points_y, 'r^', label="Próbkowane punkty")
        matplotlib.pyplot.gca().legend()
        matplotlib.pyplot.show()

    def set_random_with_zeros(self, n_points, max_factor):
        self.points_x = []
        self.points_y = []
        self.period = rand.uniform(10, 50)
        n = rand.randrange(1, 10)
        self.factors = [0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0]
        for i in range(0,n):
            p = rand.randint(0, len(self.factors)-1)
            self.factors[p] = rand.uniform(-max_factor, max_factor)
        for i in range(0, n_points):
            x = (i * self.period)/self.n_points
            self.points_x.append(x)
        self.points_x.sort()
        for i in range(0, n_points):
            y = self.function(self.points_x[i])
            self.points_y.append(y)

    def set_random_single(self, n_points, max_factor):
        self.points_x = []
        self.points_y = []
        self.period = rand.uniform(10, 50)
        p = rand.randrange(0, len(self.factors))
        self.factors = [0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0]
        self.factors[p] = rand.uniform(-max_factor, max_factor)
        for i in range(0, n_points):
            x = (i * self.period)/self.n_points
            self.points_x.append(x)
        self.points_x.sort()
        for i in range(0, n_points):
            y = self.function(self.points_x[i])
            self.points_y.append(y)

    def set_from_input_data(self, input_data):
        with open(input_data) as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                self.points_x.append(row[0])
                self.points_y.append(row[1])

    def clear(self):
        self.period = 1
        self.n_points = 100
        self.points_x = []
        self.points_y = []
        self.factors = [0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0]
        self.av_y = 0
        self.av_x = 0

    def averagize(self):
        self.av_y = sum(self.points_y)/len(self.points_y)
        for i in range(len(self.points_y)):
            self.points_y[i] = self.points_y[i] - self.av_y

    def deaveragize(self):
        for i in range(len(self.points_y)):
            self.points_y[i] = self.points_y[i] + self.av_y
        self.av_y = 0

    def set_flat(self, n_points):
        self.points_x = []
        self.points_y = []
        self.period = rand.uniform(10, 50)
        self.factors = [0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0]
        for i in range(0, n_points):
            x = (i * self.period)/self.n_points
            self.points_x.append(x)

        for i in range(0, n_points):
            y = self.function(self.points_x[i])
            self.points_y.append(y)

    def print(self):
        print(self.factors)
        print(self.period)


if __name__ == "__main__":
    pass
    case = TestCase()
    case.set_random(100, 200)
    case.show()

    # for i in range(200000):
    #     case.set_random(100, 100)
    #     case.save_to_csv('dataset.csv')
    #
    # for i in range(200000):
    #     case.set_random(100, 50)
    #     case.save_to_csv('dataset.csv')
    #
    # for i in range(200000):
    #     case.set_random_single(100, 200)
    #     case.save_to_csv('dataset.csv')
    #
    # for i in range(200000):
    #     case.set_random_single(100, 100)
    #     case.save_to_csv('dataset.csv')
    #
    # for i in range(200000):
    #     case.set_random_with_zeros(100, 200)
    #     case.save_to_csv('dataset.csv')
    #
    # for i in range(200000):
    #     case.set_random_with_zeros(100, 100)
    #     case.save_to_csv('dataset.csv')

