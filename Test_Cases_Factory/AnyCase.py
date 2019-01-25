import numpy as np

class AnyCase:
    def __init__(self, period):

        self.n_points = 100
        self.period = period
        self.points_x = np.linspace(0.1, self.period, self.n_points)
        self.points_y = np.linspace(0.1, self.period, self.n_points)
        self.function = lambda x : x
        self.av_y = 0

    def set_function(self, function):
        self.function = function
        for i in range(self.n_points):
            self.points_y[i] = self.function(self.points_x[i])

        self.av_y = sum(self.points_y)/len(self.points_y)