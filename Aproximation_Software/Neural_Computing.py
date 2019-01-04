import tensorflow as tf
from Test_Cases_Factory import Test_Case, AnyCase
import numpy as np
import matplotlib

n_input_nodes = 200
n_hidden_nodes = [600, 500]
n_output_nodes = 41

x = tf.placeholder('float', [None, n_input_nodes])
y = tf.placeholder('float')
hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_input_nodes, n_hidden_nodes[0]], stddev=0.01)),
                  'biases': tf.Variable(tf.random_normal( [n_hidden_nodes[0]], stddev=0.01))}
hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_hidden_nodes[0], n_hidden_nodes[1]], stddev=0.01)),
                  'biases': tf.Variable(tf.random_normal([n_hidden_nodes[1]], stddev=0.01))}
output_layer = {'weights': tf.Variable(tf.random_normal([n_hidden_nodes[1], n_output_nodes], stddev=0.01)),
                'biases': tf.Variable(tf.random_normal([n_output_nodes], stddev=0.01))}


def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']),  hidden_1_layer['biases'])
    l1 = tf.nn.leaky_relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']),  hidden_2_layer['biases'])
    l2 = tf.nn.leaky_relu(l2)
    output = tf.add(tf.matmul(l2,  output_layer['weights']),  output_layer['biases'])
    return output


saver = tf.train.Saver()


def function(x):
    return np.cos(x)


def test_model(model_path, test_case):
    prediction = neural_network_model(x)

    case1 = Test_Case.TestCase()
    case1.set_from_csv(test_case)

    # case1 = AnyCase.AnyCase(25)
    # case1.set_function(function)

    case2 = Test_Case.TestCase()

    features = []
    for i in range(case1.n_points):
        features.append(case1.points_x[i])
        features.append(case1.points_y[i])
    features = np.array(features)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)
        result = sess.run(prediction, feed_dict={x: [features]})

    case2.set_from_result(result[0][0], result[0][1:], case1.points_x, case1.points_y)

    l1 = np.linspace(0, case1.period, 201)
    l2 = np.linspace(0, case2.period, 201)
    matplotlib.pyplot.title('Function recovery')
    matplotlib.pyplot.subplot(2, 1, 1)
    matplotlib.pyplot.plot(l1, case1.function(l1), 'b', case1.points_x, case1.points_y, 'r^')
    matplotlib.pyplot.ylabel('Original plot')
    matplotlib.pyplot.subplot(2, 1, 2)
    matplotlib.pyplot.plot(l2, case2.function(l2), 'b', case1.points_x, case1.points_y, 'r^')
    matplotlib.pyplot.ylabel('Approximated plot')
    matplotlib.pyplot.show()


test_model('../Network_School/models2/model2/model.ckpt', 'simplesinuscase.csv')
