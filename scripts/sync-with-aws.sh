#!/bin/bash

echo "syncing raw midi files..."
aws s3 sync s3://piano-learning-stream/midi-files /var/tmp/pls/data/midi-files --delete
