#include <iostream>
#include <map>
#include <boost/filesystem.hpp>
#include <algorithm>
using namespace std;
namespace fs = ::boost::filesystem;
#include "json.hpp"
using json = nlohmann::json;


#ifndef PIANOLEARNINGEARS_STREAMHELPERS_H
#define PIANOLEARNINGEARS_STREAMHELPERS_H

map<string, vector<float>> loadSamplesIntoMemory(string ivyLocation) {
    // probably should return a pointer reference
    fs::recursive_directory_iterator it(ivyLocation);
    fs::recursive_directory_iterator endit;

    map<string, vector<float>> ret;

    int total = 880;
    int i = 0;
    while(it != endit) {

        // print progress
        if (i % (total / 10) == 0) {
            cout << 100.0 * ((float) i / (float) total) << "% done loading wavs" << endl;
        }
        string fullFilename = it->path().string();
        SNDFILE* sndfile;
        SF_INFO sfinfo;
        sndfile = sf_open(fullFilename.c_str(), SFM_READ, &sfinfo);
        if (sndfile == 0) {
            cerr << "could not open audio file " << fullFilename << endl;
            exit(1);
        }
        auto* audioIn = new float[sfinfo.channels * sfinfo.frames];
        sf_read_float(sndfile, audioIn, sfinfo.channels * sfinfo.frames);
        string keyName = it->path().stem().string();
        vector<float> audioFileAsVector;
        audioFileAsVector.insert(audioFileAsVector.end(), &audioIn[0], &audioIn[sfinfo.channels * sfinfo.frames]);
        ret[keyName] = audioFileAsVector;

        delete[] audioIn;
        sf_close(sndfile);
        i++;
        ++it;
    }

    return ret;
}

class BufferEvent {
public:
    int pianoNoteNum;
    float velocity;
    int sampleStartIndex;
    int sampleEndIndex;
    int offsetStartIndex;
};

class InputLabelPairing {
public:
    vector<float> fftInput;
    vector<float> ampLabel;
};

vector<vector<BufferEvent>> processOneJsonFile(json j) {

    vector<vector<BufferEvent>> vectorOfVectorBufferEvents;

    for (json::iterator jsoniterator = j.begin(); jsoniterator != j.end(); ++jsoniterator) {

        vector<BufferEvent> vectorOfBufferEvents;
        for (auto event: jsoniterator.value()) {
            BufferEvent bufferEvent;
            bufferEvent.pianoNoteNum = event["pianoNoteNum"];
            bufferEvent.sampleEndIndex = event["sampleEndIndex"];
            bufferEvent.sampleStartIndex = event["sampleStartIndex"];
            bufferEvent.offsetStartIndex = event["offsetStartIndex"];
            bufferEvent.velocity = event["velocity"];
            vectorOfBufferEvents.push_back(bufferEvent);
        }
        vectorOfVectorBufferEvents.push_back(vectorOfBufferEvents);
    }

    return vectorOfVectorBufferEvents;
}

vector<vector<BufferEvent>> loadMidiJsonIntoMemory(string jsonFolder) {

    fs::recursive_directory_iterator it(jsonFolder);
    fs::recursive_directory_iterator endit;

    vector<vector<BufferEvent>> vectorOfVectorBufferEvents;
    int i = 0;

    while(it != endit && i < 4) {

        string fullFilename = it->path().string();
        cout << "loading json midi file: " << fullFilename << endl;
        ifstream input(fullFilename);
        json j;
        input >> j;
        // start debugging here... fix the issue then commit...
        vector<vector<BufferEvent>> oneFile = processOneJsonFile(j);
        vectorOfVectorBufferEvents.insert(vectorOfVectorBufferEvents.end(), oneFile.begin(), oneFile.end());
        i++;
        ++it;
    }
    return vectorOfVectorBufferEvents;
}

vector<int> pickRandomBatchIndexes(int totalSize, int batchSize) {
    vector<int> arange;
    for (int i = 0; i < totalSize; i++) {
        arange.push_back(i);
    }
    random_shuffle(begin(arange), end(arange));
    vector<int> randomInts;
    for (int i = 0; i < batchSize; i++) {
        randomInts.push_back(arange[i]);
    }
    return randomInts;
}

InputLabelPairing processEvents(vector<BufferEvent>) {
    // make the signal for that buffer by getting all events relative to the offset
    // instead of wavStartIndex
}

#endif //PIANOLEARNINGEARS_STREAMHELPERS_H
