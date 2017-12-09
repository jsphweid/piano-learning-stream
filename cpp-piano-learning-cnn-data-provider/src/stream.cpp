#include <sndfile.hh>
#include "helpers/streamHelpers.h"
using namespace std;
namespace fs = ::boost::filesystem;


int main (int argc, char **argv) {

    string jsonFileLocation = "/Users/josephweidinger/Downloads/dum/";

//    // load all 880 waves in memory, as map (string, key, is filename)
    map<string, vector<float>> allSamplesInMemory = loadSamplesIntoMemory();
//    // load all json into memory... as maps (keep array of
    vector<map<int, vector<BufferEvent>>> allMidiJsonFiles = loadMidiJsonIntoMemory(jsonFileLocation);



}