#!/bin/bash

set -e

parentPath=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

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
    mkdir $outputDirectory
fi

# ----------------------------------------------------------------

echo "deleting old JSON files..."

shopt -s nullglob
sourceDirFilenames=(${midiSourceDirectory}*.mid)
sourceDirFilenames=("${sourceDirFilenames[@]##*/}") # remove path
sourceDirFilenames=("${sourceDirFilenames[@]%.mid}") # remove extensions

outputDirectoryFilenames=(${outputDirectory}*.json)
outputDirectoryFilenames=("${outputDirectoryFilenames[@]##*/}") # remove path
outputDirectoryFilenames=("${outputDirectoryFilenames[@]%.json}") # remove extensions

numOldJsonsDeleted=0

for outputDirectoryFilename in "${outputDirectoryFilenames[@]}"; do
    if [[ ! " ${sourceDirFilenames[@]} " =~ " ${outputDirectoryFilename} " ]]; then
        echo "deleting file $outputDirectoryFilename"
        rm -f "$outputDirectory/$outputDirectoryFilename.json"
        ((++numOldJsonsDeleted))
    fi
done

echo "    $numOldJsonsDeleted old json file(s) deleted."

# ----------------------------------------------------------------

echo "converting MIDI files in ${midiSourceDirectory} meaningful json..."

numJsonFilesAlreadyExist=0
numJsonFilesConverted=0
for fullFileName in ${midiSourceDirectory}*.mid; do
	fullFileNameWithoutSpaces=${fullFileName// /-}
    targetOutput=${outputDirectory}$(basename ${fullFileNameWithoutSpaces%.*}).json
    if [[ -f ${targetOutput} ]]; then
        ((++numJsonFilesAlreadyExist))
        continue;
    fi
    ((++numJsonFilesConverted))
    echo "  converting ${fullFileName} -> ${targetOutput}"
    node ./scripts/MidiConvertWrapperAdvanced.js "${fullFileName}" ${targetOutput} ${bufferSize} ${samplingRate}
done

echo "    $numJsonFilesConverted json file(s) converted."
echo "    $numJsonFilesAlreadyExist json file(s) already existed."

