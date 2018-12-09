import tensorflow as tf


class NeuralNetwork:

    def __init__(self):
        self.n_input_nodes = 100
        self.n_hidden_nodes = [250, 250]
        self.n_output_nodes = 20

        self.x = tf.placeholder('float', [None, 100])
        self.y = tf.placeholder('float')

        self.hidden_1_layer = {'weights': tf.Variable(tf.random_normal([self.n_input_nodes, self.n_hidden_nodes[0]])),
                               'biases': tf.Variable(tf.random_normal([self.n_hidden_nodes[0]]))}

        self.hidden_2_layer = {'weights': tf.Variable(tf.random_normal([self.n_hidden_nodes[0], self.n_hidden_nodes[1]])),
                               'biases': tf.Variable(tf.random_normal([self.n_hidden_nodes[1]]))}

        self.output_layer = {'weights': tf.Variable(tf.random_normal([self.n_hidden_nodes[1], self.n_output_nodes])),
                             'biases': tf.Variable(tf.random_normal([self.n_output_nodes]))}

    def graph_model(self, data):
        l1 = tf.add(tf.matmul(data, self.hidden_1_layer['weights']),  self.hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1, self.hidden_2_layer['weights']),  self.hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)

        output = tf.add(tf.matmul(l2,  self.output_layer['weights']),  self.output_layer['biases'])

        return output
