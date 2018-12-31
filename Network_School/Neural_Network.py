import tensorflow as tf
import numpy as np
import csv, random

n_input_nodes = 200
n_hidden_nodes = [500, 500]
n_output_nodes = 21
batchsize = 2000

x = tf.placeholder('float', [None, n_input_nodes])
y = tf.placeholder('float')

hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_input_nodes, n_hidden_nodes[0]], stddev=0.01)),
                  'biases': tf.Variable(tf.random_normal( [n_hidden_nodes[0]], stddev=0.01))}
hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_hidden_nodes[0], n_hidden_nodes[1]], stddev=0.01)),
                  'biases': tf.Variable(tf.random_normal([n_hidden_nodes[1]], stddev=0.01))}
output_layer = {'weights': tf.Variable(tf.random_normal([n_hidden_nodes[1], n_output_nodes], stddev=0.01)),
                'biases': tf.Variable(tf.random_normal([n_output_nodes], stddev=0.01))}


def neural_network_model(data):

    print(hidden_1_layer['weights'].shape)
    print(data.shape)

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']),  hidden_1_layer['biases'])
    l1 = tf.nn.leaky_relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']),  hidden_2_layer['biases'])
    l2 = tf.nn.leaky_relu(l2)

    output = tf.add(tf.matmul(l2,  output_layer['weights']),  output_layer['biases'])

    return output


def prepare_dataset(file_path):
    f_x = []
    f_y = []
    with open(file_path) as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)

        for row in reader:
            f_y.append(np.array(row[:21]))
            f_x.append(np.array(row[21:(21+n_input_nodes)]))
    factors = np.array(f_y)
    features = np.array(f_x)

    randomize = np.arange(len(factors))
    np.random.shuffle(randomize)
    factors = factors[randomize]
    features = features[randomize]

    return features, factors


def split_dataset(f_x, f_y, test_ratio):
    print(len(f_x)==len(f_y))
    test_size = int(test_ratio*len(f_x))
    train_x = f_x[:-test_size][:]
    train_y = f_y[:-test_size][:]
    test_x = f_x[-test_size:][:]
    test_y = f_y[-test_size:][:]
    return train_x, train_y, test_x, test_y


f_x, f_y = prepare_dataset('dataset4.csv')
train_x, train_y, test_x, test_y = split_dataset(f_x, f_y, 0.01)
saver = tf.train.Saver()


def train_neural_network(x):


    prediction = neural_network_model(x)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    cost = tf.reduce_mean(tf.square(prediction - y))
    # cost = tf.reduce_sum(tf.square(prediction - y))

    optimizer = tf.train.AdamOptimizer(learning_rate=1.0e-5).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
    hm_epochs = 200
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './models/model_81_88pr/model.ckpt')
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batchsize

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batchsize

            session_chkpt = saver.save(sess, "./models//model/model.ckpt")
            print('Epoch', epoch + 1 , 'completed out of', hm_epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

train_neural_network(x)
