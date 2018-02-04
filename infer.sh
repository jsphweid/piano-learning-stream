#!/bin/bash

wavFile=$1

if ! [ -f  $wavFile ] || [[ -z "${wavFile// }" ]]; then
    echo "must pass a .wav file to infer on as first argument..."
    exit 1
fi

# -----------------------------

checkpointsDir="/var/tmp/pls/checkpoints/"

if [ -z "$(ls -A ${checkpointsDir})" ]; then
    echo "There are no models. Please run ./train.sh first before you infer!"
    exit 1
fi

shopt -s nullglob
checkpointFiles=(${checkpointsDir}*)
checkpointFiles=("${checkpointFiles[@]##*/}") # remove path
checkpointFiles=( $( for i in ${checkpointFiles[@]} ; do echo $i ; done | grep meta ) )
checkpointFiles=("${checkpointFiles[@]%.meta}") # remove meta

for i in "${!checkpointFiles[@]}"; do 
    printf "%s\t%s\n" "$i" "${checkpointFiles[$i]}"
done

echo "which index did you want as your model?"
while true; do
    read number
    if ! [[ $number =~ ^[0-9]+$ ]] ; then
        echo "please enter a valid selection..."
        continue
    fi
    if (( ${#checkpointFiles[number]} == 0 )) ; then
        echo "please enter a valid selection..."
        continue
    fi
    echo "You chose ${checkpointFiles[number]}"
    break
done

model=${checkpointsDir}${checkpointFiles[number]}

# -----------------------------

echo "infering model..."
python ./scripts/infer.py $wavFile $model
