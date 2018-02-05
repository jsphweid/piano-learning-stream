#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <sndfile.hh>
#include "helpers/streamHelpers.h"
using namespace std;
namespace fs = ::boost::filesystem;

namespace py = pybind11;

class PianoLearnerDataProvider {
public:
    vector<vector<vector<float>>> getTrainingBatch(int batchSize);
    vector<vector<vector<float>>> getMiniTestData();
    int BUFFER_SIZE = 1024;
    float TRAINING_SIZE = 0.8;

    PianoLearnerDataProvider() {
        // load things into memory
        allSamples = loadSamplesIntoMemory("/var/tmp/pls/ivy/");
        map<string, int> sampleSizes = determineSampleSizes(allSamples);
        vectorOfVectorBufferEvents = loadMidiJsonIntoMemory("/var/tmp/pls/data/json-files/", sampleSizes);

        // get index arrays
        auto numTotalBuffers = (int) vectorOfVectorBufferEvents.size();
        randomIndexes = generateRandomIndexes(numTotalBuffers);
        auto numTrainingBuffers = (int) round(numTotalBuffers * TRAINING_SIZE);

        for (int i = 0; i < numTrainingBuffers; i++) {
            trainingIndexes.push_back(randomIndexes[i]);
        }
        for (int i = numTrainingBuffers; i < numTotalBuffers; i++) {
            testIndexes.push_back(randomIndexes[i]);
        }
        miniTestIndexes = pickRandomIndexes(testIndexes, 500);
    }
private:
    map<string, vector<float>> allSamples;
    vector<vector<BufferEvent>> vectorOfVectorBufferEvents;
    vector<int> randomIndexes;
    vector<int> trainingIndexes;
    vector<int> testIndexes;
    vector<int> miniTestIndexes;
};

vector<vector<vector<float>>> PianoLearnerDataProvider::getTrainingBatch(int batchSize) {
    vector<vector<vector<float>>> allInputsAndLabelsForBatch;

    vector<int> thisBatchRandomIndexes = pickRandomIndexes(trainingIndexes, batchSize);

    vector<vector<float>> inputs;
    vector<vector<float>> labels;

    for (auto & i : thisBatchRandomIndexes) {
        InputLabelPairing pair = processEvents(allSamples, vectorOfVectorBufferEvents[i], BUFFER_SIZE);
        inputs.push_back(pair.fftInput);
        labels.push_back(pair.ampLabel);
    }

    allInputsAndLabelsForBatch.push_back(inputs);
    allInputsAndLabelsForBatch.push_back(labels);

   return allInputsAndLabelsForBatch;
};

vector<vector<vector<float>>> PianoLearnerDataProvider::getMiniTestData() {
    vector<vector<vector<float>>> allInputsAndLabelsForTestData;
    vector<vector<float>> inputs;
    vector<vector<float>> labels;

    for (auto & i : miniTestIndexes) {
        InputLabelPairing pair = processEvents(allSamples, vectorOfVectorBufferEvents[i], BUFFER_SIZE);
        inputs.push_back(pair.fftInput);
        labels.push_back(pair.ampLabel);
    }

    allInputsAndLabelsForTestData.push_back(inputs);
    allInputsAndLabelsForTestData.push_back(labels);

    return allInputsAndLabelsForTestData;
};


PYBIND11_MODULE(cpp_piano_learning_cnn_data_provider, m) {
    m.doc() = R"pbdoc(
        Piano Learning Stream Data Provider
        -----------------------

        .. currentmodule:: cpp_piano_learning_cnn_data_provider

        .. autosummary::
           :toctree: _generate

           getTrainingBatch
           getMiniTestData
    )pbdoc";

    py::class_<PianoLearnerDataProvider> pianoLearnerDataProvider(m, "PianoLearnerDataProvider");
    pianoLearnerDataProvider.def(py::init<>());
    pianoLearnerDataProvider.def("getTrainingBatch", &PianoLearnerDataProvider::getTrainingBatch);
    pianoLearnerDataProvider.def("getMiniTestData", &PianoLearnerDataProvider::getMiniTestData);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
