import numpy as np
from numpy import reshape
import tensorflow as tf
import wave
import math
from scipy.fftpack import fft

FFT_SIZE = 512
CONV_SIZE = 20
NUM_KEYS = 88

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

saver = tf.train.Saver()

def normalizeIntArrayToOnes(int_array):
    number_of_bits = 0
    if int_array.dtype == 'int16':
        number_of_bits = 16  # -> 16-bit wav files
    elif int_array.dtype == 'int32':
        number_of_bits = 32  # -> 32-bit wav files
    if number_of_bits == 16 or number_of_bits == 32:
        max_number = float(2 ** (number_of_bits - 1))
        return int_array / (max_number + 1.0)

def getWavFileNormalizedToOnes(wavFile):
    wavFileAsIntArray = np.fromstring(wavFile.readframes(-1), 'Int16')
    return normalizeIntArrayToOnes(wavFileAsIntArray)


# load wav file and get contents
wave_file = wave.open('./wavs/bachCMajor.wav', 'r')
num_samples = wave_file.getnframes()
buffer_length = 1024
fft_length = int(buffer_length / 2)
num_even_buffers = math.floor(num_samples / buffer_length)
all_samples = getWavFileNormalizedToOnes(wave_file)

# numpy matrix for the entire size of the file

with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())
	saver.restore(sess, "/tmp/pls_checkpoints/200000-0.38111piano-learning-stream.ckpt")
	predictions = []

	for i in range(num_even_buffers):

		start_index = i * buffer_length
		this_buffer_signal = all_samples[start_index:start_index + buffer_length]
		this_buffer_fft = abs(fft(this_buffer_signal))[0:fft_length]
		this_buffer_fft_reshaped = this_buffer_fft.reshape((1, fft_length))
		raw_prediction = sess.run(y_conv, feed_dict={ x_: this_buffer_fft_reshaped.astype(float), keep_prob: 1.0 })
		predictions.append(raw_prediction)
	
	print(predictions)


	

	


