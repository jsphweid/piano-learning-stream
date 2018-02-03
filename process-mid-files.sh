#!/usr/bin/env bash

set -e

parentPath=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
projectBinariesFolder="${parentPath}/bin/"

bufferSize=1024
samplingRate=44100


# assigning arguments to better variable names
midiSourceDirectory=$1
outputDirectory=$2

# validate those arguments
if [[ $# -ne 2 ]]; then
    echo "You must supply exactly two arguments: the midi source directory and the json ouput directory."
    exit 1
fi

if [[ ! -d ${midiSourceDirectory} ]]; then
    echo "The directory ${midiSourceDirectory} does not exist."
    exit 1
elif [[ ! -d ${outputDirectory} ]]; then
    echo "The directory ${outputDirectory} does not exist."
    exit 1
fi

echo "---------------------------------------------------------------------------------"
echo "------> Converting MIDI files in ${midiSourceDirectory} to raw json."
echo "---------------------------------------------------------------------------------"

for fullFileName in ${midiSourceDirectory}*.mid; do
	fullFileNameWithoutSpaces=${fullFileName// /-}
    targetOutput=${outputDirectory}$(basename ${fullFileNameWithoutSpaces%.*}).json
    echo "-Converting ${fullFileName} ---------> ${targetOutput}"
    node ${projectBinariesFolder}MidiConvertWrapperAdvanced.js "${fullFileName}" ${targetOutput} ${bufferSize} ${samplingRate}
done
