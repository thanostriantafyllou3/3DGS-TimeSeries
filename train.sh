#!/bin/bash

## CONFIGURATION
# source directory of the dataset created by 'train_test_split.py'
SOURCE_DIR="/home/thanostriantafyllou/GS4Time/data/time_series/chair"
# number of iterations to train the model
ITERATIONS=100000
# save the model at the following iterations (e.g. to be used for rendering)
SAVE_ITERATIONS="7000 30000 100000"

# Execute the command
python train.py \
    -s "$SOURCE_DIR" \
    --eval \
    --iterations "$ITERATIONS" \
    --save_iterations $SAVE_ITERATIONS