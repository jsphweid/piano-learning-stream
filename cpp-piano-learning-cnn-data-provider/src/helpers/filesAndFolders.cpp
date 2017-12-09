#include <iostream>
#include "filesAndFolders.h"
#include <boost/filesystem.hpp>
using namespace std;
namespace fs = ::boost::filesystem;

bool tmpFolderIsFull() {
    fs::recursive_directory_iterator it("/var/tmp/ivy");
    fs::recursive_directory_iterator endit;
    int count = 0;
    while(it != endit) {
        if (fs::is_regular_file(*it) && it->path().extension() == ".wav") {
            ++count;
        }
        ++it;
    }
    return count == 880;
}
