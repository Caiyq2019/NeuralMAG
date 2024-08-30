#!/usr/bin/sh

# Temporarily set PYTHONPATH to include the top-level directory
export PYTHONPATH=$(dirname $(dirname $(dirname $(pwd)))):$PYTHONPATH

for size in 32 40 48 56 64 72 80 88 96 104 112 120 128
do
    python ./compute_vortex_fft.py --w $size \
                                   --layers 2 \
                                   --krn 16 \
                                   --split $(($size/2)) \
                                   --error_min 1.0e-5 \
                                   --dtime 5.0e-13 \
                                   --max_iter 150000 \
                                   --nsamples 800

    python ./analyze_vortex.py --w $size \
                               --split $(($size/2)) \
                               --method fft \
                               --errorfilter 1e-4

    python ./compute_vortex_unet.py --w $size \
                                    --layers 2 \
                                    --krn 16 \
                                    --split $(($size/2)) \
                                    --error_min 1.0e-5 \
                                    --dtime 5.0e-13 \
                                    --max_iter 150000 \
                                    --InitCore 10 \
                                    --nsamples 800

    python ./analyze_vortex.py --w $size \
                               --split $(($size/2)) \
                               --method unet \
                               --errorfilter 1e-4 \
                               --InitCore 10
done

python ./plot_phase_diagram.py --method fft
python ./plot_phase_diagram.py --method unet
