#!/bin/bash

# Temporarily set PYTHONPATH to include the top-level directory
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

# Define common parameters
gpu=0
nseeds=100
max_iter=50000

# Run the Python script for different combinations of width, Hext_val, and mask
for w in 32 64 96 128; do
    # When Hext_val is 0, set mask to False
    python -m utils.gen_data \
        --w $w \
        --Hext_val 0 \
        --nseeds $nseeds \
        --mask 'False' \
        --gpu $gpu \
        --max_iter $max_iter

    # When Hext_val is 100 or 1000, set mask to True
    for Hext_val in 100 1000; do
        python -m utils.gen_data \
            --w $w \
            --Hext_val $Hext_val \
            --nseeds $nseeds \
            --mask 'True' \
            --gpu $gpu \
            --max_iter $max_iter
    done
done
