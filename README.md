# Brief Description
This is an experiment with making a model that can transcribe piano music.
The model takes in an FFT slice that is [512] values in length (taken from an audio buffer that is [1024] in length) and produces its best guess as to contents (assuming the content is purely solo piano). The guess is an array of [88] numbers (with a yet-to-be-calculated range) that approximate an amplitude-like value for each key (for example: [0, 0, 0, 0, 1.5841, 0, 0.0458, ... 0, 0, 0, 0, 0, 0] means it guessed two notes are playing, one loud, one very soft).

#### A little more detail if you care
This repository contains two main components.
 - Custom C/C++ Python module (using pybind11) provides training and test data on demand.
        - stores 880 piano samples (Ivy Audio's free piano library: Piano in 162) in memory
        - processes a handful of specially tailored midi-json files [(MidiConvertWrapperAdvanced)](https://github.com/jsphweid/MidiConvertWrapperAdvanced) into many vectors containing one or more Events that 'would be' in one audio buffer memory and splits them into training/test portions
        - when `data_provider.getTrainingBatch(50)` is called (for example), a random batch of 50 arrays in the training portion is sent through a synthesizing process. This process creates 50 signals from which 50 vectors of FFTs and 50 vectors of amplitude (found by simply summing the squares of the relevant samples pertaining to each Event).
 - basic Convolutional Neural Network built with TensorFlow

# Getting it to run
 - preparing the samples
        - download here: http://ivyaudio.com/Piano-in-162
        - extract all 880 'pedal off close' samples to `/var/tmp/pls/ivy` so that it looks like:
```
/var/tmp/pls/ivy/01-PedalOffForte1Close.wav
/var/tmp/pls/ivy/01-PedalOffForte2Close.wav
...
/var/tmp/pls/ivy/88-PedalOffPiano1Close.wav
/var/tmp/pls/ivy/88-PedalOffPiano2Close.wav
```
 - dependencies
        - AWS CLI, tensorflow, python, node
        - I highly recommend creating a virtual environment... like:
```
virtualenv --system-site-packages -p python3 ~/whatever-you-want
source ~/whatever-you-want/bin/activate
pip3 install --upgrade tensorflow 
cd path/to/this/repo
pip install ./cpp-piano-learning-cnn-data-provider/ --upgrade
```
 - train
        - run `./train.sh`
 - infer
        - run `./infer.sh path/to/wav/that/you/want/to/infer.wav`

For better performance:
I'd also suggest pip installing a locally compiled version of tensorflow (https://www.tensorflow.org/install/install_sources). It is somewhat involved but it'll most likely make this train quicker. Plus you won't get a warning about it in the console.
