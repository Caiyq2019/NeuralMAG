#!/usr/bin/sh

# Temporarily set PYTHONPATH to include the top-level directory
export PYTHONPATH=$(dirname $(dirname $(dirname $(pwd)))):$PYTHONPATH

python ./MH_unet_mm.py --gpu 0 \
                        --layers 2 \
                        --w 64 \
                        --Ms 1000 \
                        --Ax 0.5e-6 \
                        --Ku 0.0 \
                        --dtime 5.0e-13 \
                        --max_iter 50000 \
                        --mask Triangle

