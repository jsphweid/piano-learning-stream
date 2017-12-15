import cpp_piano_learning_cnn_data_provider as provider
dataProvider = provider.PianoLearnerDataProvider()



#################### Adding these two lines because tensorflow wasn't compiled on this machine (used pip install)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
####################

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

base_dir = '/tmp/tensorflow/whichString/v0.03__512_fft-size/'
import numpy as np

np.set_printoptions(threshold=np.nan)

# in the future, figure out a better way of doing this shit
# https://www.tensorflow.org/programmers_guide/datasets

FFT_SIZE = 512
CONV_SIZE = 5
NUM_KEYS = 88


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

# h_pool2_reshaped = tf.reshape(h_pool2, [-1, THIRD_LAYER_SIZE])
# y_conv = tf.matmul(h_pool2_reshaped, W_fc) + b_fc
difference = tf.subtract(y_, y_conv)
squared_difference = tf.square(difference)
loss_op = tf.reduce_sum(squared_difference)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss_op)

def datasetHasBogusNumbers(dataset):
	for i in range(len(dataset)):
		if sum(dataset[i]) == 0 or sum(dataset[i]) > 20000:
			return True
	return False		

def dataprovider_wrapper(batch_size):
	## temp until we find the source of these nans...
	nan_found = True
	i = 0
	while nan_found:
		i = i + 1
		batch_xs, batch_ys = dataProvider.getTrainingBatch(batch_size)
		if not np.isnan(batch_xs).any() and not np.isnan(batch_ys).any():
			if not datasetHasBogusNumbers(batch_xs) and not datasetHasBogusNumbers(batch_ys):
				nan_found = False
	# print("went through", i, "times")
	return [batch_xs, batch_ys]



with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())

	for i in range(50000):
		# input("Press Enter to continue...")
		batch_xs, batch_ys = dataprovider_wrapper(20)
		if (np.isnan(batch_xs).any() or np.isnan(batch_ys).any()):
			print("this is not supposed to happen!!!!!!")
		# print("batch_xs", batch_xs)
		# print("batch_ys", batch_ys)
		# print("batch_xs contain nan", np.isnan(batch_xs).any())
		# print("batch_ys contain nan", np.isnan(batch_ys).any())
		diff, s_diff, loss, _ = sess.run([difference, squared_difference, loss_op, train_step], feed_dict={x_: batch_xs, y_: batch_ys, keep_prob: 0.5})
		# print('diff', diff)
		# print('s_diff', s_diff)
		# print('loss', loss)

		# print('chek_numerics', chek_numerics)
		if i % 10 == 0:
			test_xs, test_ys = dataprovider_wrapper(100)
			initial_x, expected_y, end_y, test_loss = sess.run([x_, y_, y_conv, loss_op], feed_dict={x_: test_xs, y_: test_ys, keep_prob: 1.0})
			# test_loss = loss_op.eval(feed_dict={x_: test_xs, y_: test_ys, keep_prob: 1.0})
			# print('initial_x', initial_x)
			# print('expected_y', expected_y)
			# print("endy", end_y)
			print('step %d, loss from test data %g' % (i, test_loss))




