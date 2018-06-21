'''
input > weight > hidden layer 1 (activation function) > weights > hidden layer 2
(activation function) > weights > output layer

compare output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer, SGD, AdaGrad)

backpropagation

feed forward + backdrop = epoch
''' 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)  # one component will be hot then the rest are off, electricity runs through it then its hot
 # 10 classes, 0-9
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
# reality youre going to have millions of millions of data, you dont have peta bites of ram
batch_size = 100

#a matrix is height by width
x = tf.placeholder('float', [None, 784])  # input data, will have a very specific shape
y = tf.placeholder('float')  # height is zero

def neural_network_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
					  'biases':tf.Variable(tf.random_normal(n_nodes_hl1))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl])),
					  'biases':tf.Variable(tf.random_normal(n_nodes_hl2))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					  'biases':tf.Variable(tf.random_normal(n_nodes_hl3))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
					  'biases':tf.Variable(tf.random_normal([n_classes]))}