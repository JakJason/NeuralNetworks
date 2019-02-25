import tensorflow as tf
from Test_Cases_Factory import Test_Case, AnyCase
import numpy as np
import matplotlib
from tkinter import *
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


n_input_nodes = 200
n_hidden_nodes = [150, 150]
n_output_nodes = 31

x = tf.placeholder('float', [None, n_input_nodes])
y = tf.placeholder('float')

hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_input_nodes, n_hidden_nodes[0]], stddev=0.001)),
                  'biases': tf.Variable(tf.random_normal( [n_hidden_nodes[0]], stddev=0.001))}
hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_hidden_nodes[0], n_hidden_nodes[1]], stddev=0.001)),
                  'biases': tf.Variable(tf.random_normal([n_hidden_nodes[1]], stddev=0.001))}
output_layer = {'weights': tf.Variable(tf.random_normal([n_hidden_nodes[1], n_output_nodes], stddev=0.001)),
                'biases': tf.Variable(tf.random_normal([n_output_nodes], stddev=0.001))}

saver = tf.train.Saver()


def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']),  hidden_1_layer['biases'])
    l1 = tf.nn.leaky_relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']),  hidden_2_layer['biases'])
    l2 = tf.nn.leaky_relu(l2)
    output = tf.add(tf.matmul(l2,  output_layer['weights']),  output_layer['biases'])
    return output


def approximate(model_path, test_case):
    prediction = neural_network_model(x)
    features = []
    test_case[0].av_y = np.mean(test_case[0].points_y)
    for i in range(test_case[0].n_points):
        features.append(test_case[0].points_x[i])
        features.append(test_case[0].points_y[i] - test_case[0].av_y)
    features = np.array(features)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)
        result = sess.run(prediction, feed_dict={x: [features]})
    test_case[0].set_from_result(result[0][0], result[0][1:], test_case[0].av_y, test_case[0].points_x, test_case[0].points_y)


class Application:
    def __init__(self):
        self.master = Tk()
        self.master.title('Approximation Software')
        self.master.configure(background='grey')

        self.Button_Frame = Frame(highlightbackground="grey", highlightcolor="grey", highlightthickness=5)
        self.Load_Button = Button(self.Button_Frame,
                                  text='Load data',
                                  state=NORMAL,
                                  width=30,
                                  height=2,
                                  command=self.load_points)
        self.Approx_Button = Button(self.Button_Frame,
                                    text='Calculate approximation',
                                    state=NORMAL,
                                    width=30,
                                    height=2,
                                    command=self.approximate)
        self.Clear_Button = Button(self.Button_Frame,
                                   text='Clear model',
                                   state=NORMAL,
                                   width=30,
                                   height=2,
                                   command=self.clear)
        self.Load_Button.grid(row=0, column=0, sticky="nsew")
        self.Approx_Button.grid(row=1, column=0, sticky="nsew")
        self.Clear_Button.grid(row=2, column=0, sticky="nsew")

        self.Plot_Frame = Frame(highlightbackground="grey", highlightcolor="grey", highlightthickness=5)

        self.Other_Frame = Frame(highlightbackground="grey", highlightcolor="grey", highlightthickness=5)

        self.Button_Frame.grid(row=0, column=0, sticky="nsew")
        self.Plot_Frame.grid(row=0, column=1, sticky="nsew")
        self.Other_Frame.grid(row=0, column=2, sticky="nsew")

        self.fig = matplotlib.figure.Figure(figsize=(8, 8))
        self.a = None
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.Plot_Frame)
        self.canvas.get_tk_widget().grid(row=0, column=0)
        self.case = Test_Case.TestCase()

        self.Equation = StringVar()
        self.Equation_Frame = Label(self.Other_Frame, height=25, width=75, textvariable=self.Equation)
        self.Equation_Frame.config(font=('times', 12))
        self.Equation.set("S(x) = ")
        self.Equation_Frame.pack()

    def load_points(self):
        input_data = filedialog.askopenfilename(defaultextension='.csv')
        if input_data is not None and len(input_data) > 0:
            self.case.set_from_input_data(input_data)
        if self.fig is not None:
            self.fig.clf()
        self.a = self.fig.add_subplot(111)
        self.a.plot(self.case.points_x, self.case.points_y, 'r^', label="Próbkowane punkty")
        self.fig.gca().legend()
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0)

    def approximate(self):
        approximate('../Network_School/models/model 200-150-150-31 98.83pr/model.ckpt', [self.case])
        l1 = np.linspace(0.001, self.case.period, 401)
        if self.fig is not None:
            self.fig.clf()
        self.a = self.fig.add_subplot(111)
        self.a.plot(self.case.points_x, self.case.points_y, 'r^', label="Próbkowane punkty")
        self.a.plot(l1, self.case.function(l1), 'b', label="Aproksymowana funkcja")
        self.fig.gca().legend()
        self.format_equation()
        self.canvas.draw()

    def format_equation(self):

        eq = "S(x) = {0} + \n".format(self.case.av_y)

        for i in range(int(len(self.case.factors)/2)):
            eq += "+ ({0}) * cos(({1} * pi * x)/2) + ({2}) * sin(({1} * pi * x)/2) \n".format(self.case.factors[i*2], 2*(i+1), self.case.factors[i*2 + 1])

        self.Equation.set(eq)


    def clear(self):
        self.case.clear()
        self.Equation.set("S(x) = ")
        if self.fig is not None:
            self.fig.clf()
        self.canvas.draw()


def main():
    p = Application()
    p.master.mainloop()


if __name__ == '__main__':
    main()