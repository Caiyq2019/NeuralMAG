#!/usr/bin/sh

# Temporarily set PYTHONPATH to include the top-level directory
export PYTHONPATH=$(dirname $(dirname $(dirname $(pwd)))):$PYTHONPATH

for size in 32 64 96 128 256 512
do
    python ./prb4_compare.py --w $size
done

python ./prb4_compare.py --w 512 --converge True


# plot figure
python ./plot_data.py
