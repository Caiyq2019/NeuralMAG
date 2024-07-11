#!/usr/bin/sh

# Temporarily set PYTHONPATH to include the top-level directory
export PYTHONPATH=$(dirname $(dirname $(dirname $(pwd)))):$PYTHONPATH


logfile="run_standard_prbs.log"

# Run problem1
echo "start problem1" > $logfile
date >> $logfile
python prb1_fftHd.py
date >> $logfile
echo "end problem1" >> $logfile
echo "" >> $logfile

# Run problem2
echo "start problem2" >> $logfile
date >> $logfile
python prb2_fftHd.py
date >> $logfile
echo "end problem2" >> $logfile
echo "" >> $logfile

# Run problem3
echo "start problem3" >> $logfile
date >> $logfile
python prb3_fftHd.py
date >> $logfile
echo "end problem3" >> $logfile
echo "" >> $logfile

# Run problem4
echo "start problem4" >> $logfile
date >> $logfile
python prb4_fftHd.py
date >> $logfile
echo "end problem4" >> $logfile
