#include <iostream>
#include <map>
#include <boost/filesystem.hpp>
#include <algorithm>
using namespace std;
namespace fs = ::boost::filesystem;
#include "json.hpp"
using json = nlohmann::json;

#define MEOW_FFT_IMPLEMENTAION
#include "../lib-src/header-only/meow_fft.h"
#include <complex>


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
        auto* audioIn = new float[sfinfo.frames];
        sf_read_float(sndfile, audioIn, sfinfo.frames);
        string keyName = it->path().stem().string();
        vector<float> audioFileAsVector;

        audioFileAsVector.insert(audioFileAsVector.end(), &audioIn[0], &audioIn[sfinfo.frames]);
    
        if (keyName == "22-PedalOffMezzoPiano1Close") {
            // for (int i = 0; i < sfinfo.frames; i++) {
            //     cout << audioIn[i] << " ";
            // }
            // cout << endl;
            // for (int i = 0; i < audioFileAsVector.size(); i++) {
            //     cout << audioFileAsVector[i] << " ";
            // }
            // cout << endl;
            // for (float thing : audioFileAsVector) {
            //     cout << thing << " ";
            // }
            // cout << endl;
            // cout << audioFileAsVector.size() << endl;
        }
        ret[keyName] = audioFileAsVector;

        delete[] audioIn;
        sf_close(sndfile);
        i++;
        ++it;
    }

    // vector<float> sample = ret.find("22-PedalOffMezzoPiano1Close")->second;
    // for (int i : sample) {
    //     cout << i << " ";
    // }
    // cout << endl;

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

    while(it != endit && i < 4) { // temp limit is 4, take i out eventually

        string fullFilename = it->path().string();
        cout << "loading json midi file: " << fullFilename << endl;
        ifstream input(fullFilename);
        json j;
        input >> j;

        vector<vector<BufferEvent>> oneFile = processOneJsonFile(j);
        vectorOfVectorBufferEvents.insert(vectorOfVectorBufferEvents.end(), oneFile.begin(), oneFile.end());
        i++;
        ++it;
    }
    return vectorOfVectorBufferEvents;
}

vector<int> pickRandomIndexes(vector<int> vectorToPickFrom, int numToPick) {
    random_shuffle(begin(vectorToPickFrom), end(vectorToPickFrom));
    vector<int> randomInts;
    for (int i = 0; i < numToPick; i++) {
        randomInts.push_back(vectorToPickFrom[i]);
    }
    return randomInts;
}

vector<int> generateRandomIndexes(int totalSize) {
    vector<int> arange;
    for (int i = 0; i < totalSize; i++) {
        arange.push_back(i);
    }
    return pickRandomIndexes(arange, totalSize);
}

string convertMidiNoteToPianoNoteNum(int midiNote) {
    int newNum = midiNote - 20;
    string numAsString = to_string(newNum);
    return (newNum < 10) ? "0" + numAsString : numAsString;
}

string pickAppropriateWavFile(int midiNote, float velocity) {
    string levels[10] = {
            "PedalOffPiano1Close",
            "PedalOffPiano2Close",
            "PedalOffPianissimo1Close",
            "PedalOffPianissimo1Close",
            "PedalOffMezzoPiano1Close",
            "PedalOffMezzoPiano2Close",
            "PedalOffMezzoForte1Close",
            "PedalOffMezzoForte2Close",
            "PedalOffForte1Close",
            "PedalOffForte2Close"
    };

    auto index = (int) floor(velocity * 10.0);
    return convertMidiNoteToPianoNoteNum(midiNote) + "-" + levels[index];
}

void getFFTOfBuffer(int bufferSize, int fftSize, float* audioBufferIn, float* maggedArrayOut) {

    // initialize array to hold the output of the fft, before a magnitude operation is applied
    Meow_FFT_Complex* out = (Meow_FFT_Complex*) malloc(sizeof(Meow_FFT_Complex) * bufferSize);

    // prep
    size_t workset_bytes = meow_fft_generate_workset_real(bufferSize, NULL);
    Meow_FFT_Workset_Real* fft_real = (Meow_FFT_Workset_Real*) malloc(workset_bytes);
    meow_fft_generate_workset_real(bufferSize, fft_real);

    // do the transform and put it in out
    meow_fft_real(fft_real, audioBufferIn, out);

    // go through out and take the magnitude of each sample and put it in the final destination array
    for (int i = 0; i < fftSize; i++) {
        complex<float> mycomplex(out[i].r, out[i].j);
        maggedArrayOut[i] = abs(mycomplex);
    }

    // free the temp out array only used in this function
    free(out);
    free(fft_real);
}

InputLabelPairing processEvents(map<string, vector<float>> const &allSamples, vector<BufferEvent> bufferEvents, int bufferSize) {
    // cout << "-------------------" << endl;
    auto* signal = new float[bufferSize]();
    int fftSize = (int) bufferSize / 2;
    auto* fft = new float[fftSize]();
    vector<float> ampLabelVector(88, 0);

    // cout << "22-PedalOffMezzoPiano1Close.wav: ";
    // vector<float> stuff = allSamples.find("22-PedalOffMezzoPiano1Close")->second;
    // for (int const &item : stuff) {
    //     cout << item << " ";
    // }
    // cout << endl;

    for (BufferEvent bufferEvent : bufferEvents) {
        // cout << "pianoNoteNum " << bufferEvent.pianoNoteNum << endl;
        // cout << "velocity " << bufferEvent.velocity << endl;
        // cout << "sampleStartIndex " << bufferEvent.sampleStartIndex << endl;
        // cout << "sampleEndIndex " << bufferEvent.sampleEndIndex << endl;
        // cout << "offsetStartIndex " << bufferEvent.offsetStartIndex << endl << endl;;
        string sampleName = pickAppropriateWavFile(bufferEvent.pianoNoteNum, bufferEvent.velocity);
        
        vector<float> sample = allSamples.find(sampleName)->second;

        int sampleLength = bufferEvent.sampleEndIndex - bufferEvent.sampleStartIndex;
        float amp = 0;
        // cout << "sample length: " << sampleLength << endl;
        // if (sampleLength != 1024) cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << endl;
        // cout << "offset start index: " << bufferEvent.offsetStartIndex << endl;
        for (int i = bufferEvent.offsetStartIndex, j = 0; i < bufferEvent.offsetStartIndex + sampleLength; i++, j++) {
            auto currentSample = sample[bufferEvent.sampleStartIndex + j];
            signal[i] += currentSample;
            amp += (float) pow(currentSample, 2);
        }
        // cout << endl;
        // cout << "final amp for piano note: " << bufferEvent.pianoNoteNum << ": " << amp << endl;
        ampLabelVector[bufferEvent.pianoNoteNum - 1] = amp;
    }

    // for (int i = 0; i < bufferSize; i++) {
    //     cout << signal[i] << " ";
    // }
    // cout << endl;


    getFFTOfBuffer(bufferSize, fftSize, signal, fft);
    vector<float> fftAsVector;
    fftAsVector.insert(fftAsVector.end(), &fft[0], &fft[fftSize]);
    delete[] fft;
    delete[] signal;

    // for (auto item : fftAsVector) {
    //     cout << item << " ";
    // }
    // cout << endl;

    InputLabelPairing ret;
    ret.ampLabel = ampLabelVector;
    ret.fftInput = fftAsVector;
    return ret;
}

#endif //PIANOLEARNINGEARS_STREAMHELPERS_H
