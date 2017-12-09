#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <sndfile.hh>
#include "helpers/streamHelpers.h"
using namespace std;
namespace fs = ::boost::filesystem;

namespace py = pybind11;

class MyClass {
public:
    MyClass() {
        int BATCH_SIZE = 10;
        // allSamples = loadSamplesIntoMemory("/var/tmp/ivy/");
        vectorOfVectorBufferEvents = loadMidiJsonIntoMemory("/Users/josephweidinger/Downloads/dum/");
        vector<int> randomIndexes = pickRandomBatchIndexes((int) vectorOfVectorBufferEvents.size(), BATCH_SIZE);


        // TODO:
//        vector<InputLabelPairing> allInputsAndLabelsForBatch;
//
//        for (int i = 0; i < BATCH_SIZE; i++) {
//            InputLabelPairing pair = processEvents(vectorOfVectorBufferEvents[i]);
//            allInputsAndLabelsForBatch.push_back(pair);
//        }

    }
    void dostuff();

private:
    // map<string, vector<float>> allSamples;
    vector<vector<BufferEvent>> vectorOfVectorBufferEvents;
};

void MyClass::dostuff() {
    cout << "Hello from dostuff" << endl;
}

// vector<vector<BufferEvent>> MyClass::getVectorOfVectorBufferEvents() {
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

    py::class_<MyClass> myClass(m, "MyClass");
    myClass.def(py::init<>());
    myClass.def("dostuff", &MyClass::dostuff);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
