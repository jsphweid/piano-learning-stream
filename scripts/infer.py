import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import sys
import tensorflow as tf
from nn import nn
import wave
import math

WAV_FILE_PATH = sys.argv[1]
MODEL_PATH = sys.argv[2]

FFT_SIZE = 512
BUFFER_LENGTH = FFT_SIZE * 2


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
wave_file = wave.open(WAV_FILE_PATH, 'r')
num_samples = wave_file.getnframes()
num_even_buffers = math.floor(num_samples / BUFFER_LENGTH)
all_samples = getWavFileNormalizedToOnes(wave_file)

# numpy matrix for the entire size of the file

predictions = []

nnSess = nn.NNSession()
nnSess.restoreGraph(MODEL_PATH)

for i in range(400):
	start_index = i * BUFFER_LENGTH
	this_buffer_signal = all_samples[start_index:start_index + BUFFER_LENGTH]
	this_buffer_fft = abs(fft(this_buffer_signal))[0:FFT_SIZE]
	this_buffer_fft_reshaped = this_buffer_fft.reshape((1, FFT_SIZE))
	raw_prediction = nnSess.runGraph(this_buffer_fft_reshaped.astype(float), 1.0)
	predictions.append(raw_prediction[0])

def make_wave_from_intensity_and_index(index, intensity):
    frequency = 440 * (1.059463094359 ** (index - 49))
    x = np.arange(BUFFER_LENGTH)
    return np.sin(2 * np.pi * frequency * x / 44100) * intensity
    
master_signals = []

for prediction in predictions:
    signals = []
    for index, intensity in enumerate(prediction):
        signals.append(make_wave_from_intensity_and_index(index, intensity))
    master_signals.append([sum(x) for x in zip(*signals)])

final_signal = [item for sublist in master_signals for item in sublist]
normalization_factor = np.amax(final_signal)
final_signal = final_signal * (1 / normalization_factor)

import scipy.io.wavfile
scipy.io.wavfile.write("karplus.wav", 44100, final_signal)

def plot_arbitrary_2d_data_as_spectrogram(data, x_label="X", y_label="Y"):
    
    x_size, y_size = np.shape(data)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(
        np.transpose(data),
        origin="lower",
        aspect="auto",
        cmap="jet",
        interpolation="none"
    )
    plt.colorbar()

    # set up axes
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    x_spread = np.linspace(0, x_size - 1, 8)
    plt.xticks(x_spread)
    y_spread = np.linspace(0, y_size - 1, 8)
    plt.yticks(y_spread)

    plt.show()
    plt.clf()

# plot_arbitrary_2d_data_as_spectrogram(predictions, 'sample number', 'key')

