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
    PianoLearnerDataProvider() {
        float TRAINING_SIZE = 0.8;
        allSamples = loadSamplesIntoMemory("/var/tmp/ivy/");
        vectorOfVectorBufferEvents = loadMidiJsonIntoMemory("/Users/josephweidinger/Downloads/dum/");


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


    }
    vector<vector<vector<float>>> getTrainingBatch(int batchSize);
    vector<vector<vector<float>>> getRandomTestData(int batchSize);
    int BUFFER_SIZE = 1024;

private:
    map<string, vector<float>> allSamples;
    vector<vector<BufferEvent>> vectorOfVectorBufferEvents;
    vector<int> randomIndexes;
    vector<int> trainingIndexes;
    vector<int> testIndexes;
};

vector<vector<vector<float>>> PianoLearnerDataProvider::getTrainingBatch(int batchSize) {
    vector<vector<vector<float>>> allInputsAndLabelsForBatch;

    vector<int> thisBatchRandomIndexes = pickRandomIndexes(trainingIndexes, batchSize);

    ////////////// TEMP PRINT START
    // cout << "this training label batch size: " << batchSize << " of " << trainingIndexes.size() << endl;
    // cout << "training indexes for this batch indexes: ";
    // cout << endl;
    // cout << "vectorOfVectorBufferEvents size: " << vectorOfVectorBufferEvents.size() << endl;
    ////////////// TEMP PRINT END

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

vector<vector<vector<float>>> PianoLearnerDataProvider::getRandomTestData(int batchSize) {
    vector<vector<vector<float>>> allInputsAndLabelsForTestData;
    vector<int> thisBatchRandomIndexes = pickRandomIndexes(testIndexes, batchSize);
    ////////////// TEMP PRINT START
    // cout << "total test data: " << allInputsAndLabelsForTestData.size() << endl;
    // cout << "test indexes: ";
    // cout << endl;
    ////////////// TEMP PRINT END

    vector<vector<float>> inputs;
    vector<vector<float>> labels;

    for (auto & i : thisBatchRandomIndexes) {
        InputLabelPairing pair = processEvents(allSamples, vectorOfVectorBufferEvents[i], BUFFER_SIZE);
        inputs.push_back(pair.fftInput);
        labels.push_back(pair.ampLabel);
    }

    allInputsAndLabelsForTestData.push_back(inputs);
    allInputsAndLabelsForTestData.push_back(labels);

    return allInputsAndLabelsForTestData;
};

// vector<vector<BufferEvent>> PianoLearnerDataProvider::getVectorOfVectorBufferEvents() {
//     return vectorOfVectorBufferEvents;
// };



// get random batch
// 1. figure out length of all json BUFFERS (does it contain small buffer at end? hope not)
// 2. make an array of indexes from 0 to that num
// 3. shuffle that array on each cycle
// 4. get the first 0:batch size numbers
// that might be tricky
// grab those 20 files, and go to town

// training and test data
// 1 idea: get that big array of numbers from above and take out 30% of them and put them in another array...
// that will be the training set...

// the problem with both of these, is we have to associte a big list of numbers as filename -> its list
// but not just have a vector of Buffer events instead of separate files? combine them all?




PYBIND11_MODULE(cpp_piano_learning_cnn_data_provider, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cpp_piano_learning_cnn_data_provider

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    py::class_<PianoLearnerDataProvider> pianoLearnerDataProvider(m, "PianoLearnerDataProvider");
    pianoLearnerDataProvider.def(py::init<>());
    pianoLearnerDataProvider.def("getTrainingBatch", &PianoLearnerDataProvider::getTrainingBatch);
    pianoLearnerDataProvider.def("getRandomTestData", &PianoLearnerDataProvider::getRandomTestData);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
