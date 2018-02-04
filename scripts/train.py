FFT_SIZE = 512
CONV_SIZE = 20
NUM_KEYS = 88
CHECKPOINT_DIR = '/var/tmp/pls/checkpoints/'

import cpp_piano_learning_cnn_data_provider as provider
dataProvider = provider.PianoLearnerDataProvider()

import numpy as np
import tensorflow as tf 

def get_weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def get_bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv1d(x, W):
	return tf.nn.conv1d(x, W, stride=1, padding='VALID')

def max_pool_2(x):
    return tf.layers.max_pooling1d(x, 2, 2, padding='VALID') #pooling probably is affecting the wrong thing...

######### DECLARE VARIABLES
x_ = tf.placeholder(tf.float32, shape=[None, FFT_SIZE], name="x_")
y_ = tf.placeholder(tf.float32, shape=[None, NUM_KEYS], name="y_")
keep_prob = tf.placeholder(tf.float32)

x_resized = tf.reshape(x_, [-1, FFT_SIZE, 1])

######### FIRST CONVOLUTIONAL LAYER
FIRST_LAYER_SIZE = 32
W_conv1 = get_weight_variable([CONV_SIZE, 1, FIRST_LAYER_SIZE])
b_conv1 = get_bias_variable([FIRST_LAYER_SIZE])
conv1 = conv1d(x_resized, W_conv1)
h_conv1 = tf.nn.relu(conv1 + b_conv1)
h_pool1 = max_pool_2(h_conv1)

FFT_SIZE_AFTER_POOL1 = int((FFT_SIZE - CONV_SIZE + 1) / 2)

######### SECOND CONVOLUTIONAL LAYER
SECOND_LAYER_SIZE = 64
W_conv2 = get_weight_variable([CONV_SIZE, FIRST_LAYER_SIZE, SECOND_LAYER_SIZE])
b_conv2 = get_bias_variable([SECOND_LAYER_SIZE])
conv2 = conv1d(h_pool1, W_conv2)
h_conv2 = tf.nn.relu(conv2 + b_conv2)
h_pool2 = max_pool_2(h_conv2)

FFT_SIZE_AFTER_POOL2 = int((FFT_SIZE_AFTER_POOL1 - CONV_SIZE + 1) / 2)
THIRD_LAYER_SIZE = FFT_SIZE_AFTER_POOL2 * SECOND_LAYER_SIZE

######### DENSELY CONNECTED LAYER
DENSELY_CONNECTED_LAYER_SIZE = 512
W_fc1 = get_weight_variable([THIRD_LAYER_SIZE, DENSELY_CONNECTED_LAYER_SIZE])
b_fc1 = get_bias_variable([DENSELY_CONNECTED_LAYER_SIZE])
h_pool2_flat = tf.reshape(h_pool2, [-1, THIRD_LAYER_SIZE])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

######### DROPOUT LAYER
h_fc1_dropped = tf.nn.dropout(h_fc1, keep_prob)

######### READOUT LAYER
W_fc2 = get_weight_variable([DENSELY_CONNECTED_LAYER_SIZE, NUM_KEYS])
b_fc2 = get_bias_variable([NUM_KEYS])

y_conv = tf.matmul(h_fc1_dropped, W_fc2) + b_fc2

loss_op = tf.reduce_mean(tf.square(tf.subtract(y_, y_conv)))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss_op)


with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	lowest_loss = 10

	for i in range(500000):
		batch_xs, batch_ys = dataProvider.getTrainingBatch(30)

		training_loss, _ = sess.run([loss_op, train_step], feed_dict={x_: batch_xs, y_: batch_ys, keep_prob: 0.5})

		if i % 10 == 0:

			test_xs, test_ys = dataProvider.getMiniTestData()
			test_loss = loss_op.eval(feed_dict={x_: test_xs, y_: test_ys, keep_prob: 1.0})

			print(i, ':', 'loss from training', training_loss, ': loss from test', test_loss)
			
			if test_loss < lowest_loss:
				lowest_loss = test_loss
				print('--------------------------> new record at num:', lowest_loss)
				filename = CHECKPOINT_DIR + str(i) + '-' + str(test_loss) + 'piano-learning-stream.ckpt'
				save_path = saver.save(sess, filename)

