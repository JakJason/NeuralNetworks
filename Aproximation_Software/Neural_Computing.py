import tensorflow as tf
from Test_Cases_Factory import Test_Case, AnyCase
import numpy as np
import matplotlib

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


def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']),  hidden_1_layer['biases'])
    l1 = tf.nn.leaky_relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']),  hidden_2_layer['biases'])
    l2 = tf.nn.leaky_relu(l2)
    output = tf.add(tf.matmul(l2,  output_layer['weights']),  output_layer['biases'])
    return output


saver = tf.train.Saver()


def function(x):
    # return x * np.sin(3*x)
    # return 10*(x-10)/np.abs(x-10)
    # return 0.1*(x-10)**4 - (x-10)**3 - 34*((x-10)**2) - 21*(x-10) + 6
    # return (x-10)**2
    # return x
    # return 32 * np.log10(x)
    # return (2**x)/5000
    return 0.3*(x-10)**3
    # return 2*x/x
    # return 3*np.cos(2*x) + 2*np.sin(3*x)
    # return (x/x) * np.random.uniform(-50, 50)
    # return (x/x) * np.random.normal(0, 20)

def test_model(model_path, test_case):
    prediction = neural_network_model(x)

    # case1 = Test_Case.TestCase()
    # case1.set_from_csv(test_case)
    case1 = AnyCase.AnyCase(20)
    case1.set_function(function)
    case2 = Test_Case.TestCase()

    features = []
    case2.av_y = np.mean(case1.points_y)
    for i in range(case1.n_points):
        features.append(case1.points_x[i])
        features.append(case1.points_y[i] - case2.av_y)

    # s = features[-2] - features[0]
    # f1 = features[:100]
    # for i in range(50):
    #     f1[i*2] = f1[i*2] - s
    # f2 = features[100:]
    # for i in range(50):
    #     f2[i*2] = f2[i*2] + s
    # features = f1 + features + f2

    features = np.array(features)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)
        result = sess.run(prediction, feed_dict={x: [features]})

    # print(result[0][0])
    # print(case2.av_y)
    # print(result[0][1:])

    case2.set_from_result(result[0][0], result[0][1:], case1.av_y, case1.points_x, case1.points_y)
    # l1 = np.linspace(0 - case1.period/2, 3*case1.period/2, 401)
    l1 = np.linspace(0.001, case1.period, 401)


    matplotlib.pyplot.figure(1)
    # matplotlib.pyplot.subplot(3, 1, 1)
    # matplotlib.pyplot.plot(l1, case1.function(l1), 'b', label="Original function")
    # matplotlib.pyplot.plot(case1.points_x, case1.points_y, 'r^', label="Input points")
    # matplotlib.pyplot.gca().legend()
    # matplotlib.pyplot.subplot(3, 1, 2)
    matplotlib.pyplot.plot(l1, case2.function(l1), 'b', label="Aproksymowana funkcja")
    matplotlib.pyplot.plot(case1.points_x, case1.points_y, 'r^', label="dane wejściowe")
    matplotlib.pyplot.gca().legend()
    # matplotlib.pyplot.subplot(3, 1, 3)
    # matplotlib.pyplot.plot(l1, case1.function(l1) - case2.function(l1), 'b', label="Diffrence")
    # matplotlib.pyplot.gca().legend()

    n = len(case2.points_x)
    s = 0
    for i in range(0, n):
        s = s + (case1.points_y[i] - case2.function(case2.points_x[i]))**2

    print("Root Mean Square Error")
    print(np.sqrt(s/n))

    print("Rel Root Mean Square Error:")
    T = float(max(case2.points_y) - min(case2.points_y))
    rel_s = (np.sqrt(s/n))/T
    print(rel_s)

    print("Cut Root Mean Square Error:")
    for i in range(10, 80):
        s = s + (case1.points_y[i] - case2.function(case2.points_x[i]))**2
    print(np.sqrt(s/(n-20)))

    print("RC Root Mean Square Error:")
    rel_s2 = (np.sqrt(s/(n-20)))/T
    print(rel_s2)

    print(str(format(rel_s, '.5g'))+ " & " + str(format(np.sqrt(s/(n-20)), '.5g'))+ " & " + str(format(rel_s2, '.5g'))+ " \\\\ ")


    matplotlib.pyplot.figure(2)
    matplotlib.pyplot.plot(l1, case1.function(l1), 'b', label="Funkcja oryginalna f(x)")
    matplotlib.pyplot.plot(l1, case2.function(l1), 'r', label="Aproksymacja S(x)")
    matplotlib.pyplot.plot(l1, case1.function(l1) - case2.function(l1), 'g', label="Różnica f(x) - S(x)")
    matplotlib.pyplot.gca().legend()

    matplotlib.pyplot.show()


test_model('../Network_School/models/model 200-150-150-31 98.83pr/model.ckpt', 'testcase1.csv')
