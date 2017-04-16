import tensorflow as tf 
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
import pandas

dataset_path      = "/path/to/out/dataset/mnist/"
test_labels_file  = "test-labels.csv"
train_labels_file = "train-labels.csv"

inputdata = "bla"

input_width = 200
input_height = 200
channels = 3
input_fulllength = input_height*input_width*channels

n_nodes_hl1 = 10000
n_nodes_hl2 = 1000
n_nodes_hl3 = 200

n_classes = 5
batch_size = 1 #Will vary based on choice

placeholder = tf.placeholder('float',[None, input_fulllength])

def neural_network_model(data):

    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([input_fulllength, n_nodes_hl1])),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
    'biases':tf.Variable(tf.random_normal([n_classes]))}

    # The actual model starts here

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3) 
    
    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output
    
def train_neural_network(x):

    prediction = neural_network_model(x)

    cost = tf. reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    num_opochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(int(inputdata.train.num_examples/batch_size)):
                epoch_x, epoch_y = inputdata.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:inputdata.test.images, y:mnist.test.labels})) 

train_neural_network(x)