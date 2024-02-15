#!/usr/bin/sh

# Temporarily set PYTHONPATH to include the top-level directory
export PYTHONPATH=$(dirname $(dirname $(dirname $(pwd)))):$PYTHONPATH

width=3072

python ./MH_unet.py --w $width \
                    --layer 2 \
                    --Ms 1000 \
                    --Ax 0.5e-6 \
                    --Ku 0.0 \
                    --dtime 2.0e-13

