#!/bin/bash

echo "------> syncing files with AWS..."
bash ./scripts/sync-with-aws.sh

echo -e "\n------> preprocessing midi files..."
bash ./scripts/process-mid-files.sh /var/tmp/pls/data/midi-files/ /var/tmp/pls/data/json-files/

echo -e "\n------> training model..."
python ./scripts/train.py