#!/usr/bin/sh

# Temporarily set PYTHONPATH to include the top-level directory
export PYTHONPATH=$(dirname $(dirname $(dirname $(pwd)))):$PYTHONPATH


size=32

for pre_core in 5; do
    python ./compute_vortex.py --w $size \
                               --layers 2 \
                               --krn 16 \
                               --split 32 \
                               --pre_core $pre_core \
                               --error_min 1.0e-5 \
                               --dtime 5.0e-13 \
                               --max_iter 50000 \
                               --nsamples 10

    python ./analyze_vortex.py --w $size \
                               --krn 16 \
                               --split 32 \
                               --pre_core $pre_core \
                               --errorfilter 1e-5
done