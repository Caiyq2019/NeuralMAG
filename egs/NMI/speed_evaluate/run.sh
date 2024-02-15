#!/usr/bin/sh

# Temporarily set PYTHONPATH to include the top-level directory
export PYTHONPATH=$(dirname $(dirname $(dirname $(pwd)))):$PYTHONPATH

width=1024

# Define the log file path
LOG_FILE="speed_test_w$width.log"

# Clear the log file if it exists
> "$LOG_FILE"


# Run the Python scripts and append output to the log file
{
    python unet_speed.py --gpu 0 --w $width --layers 2 --trt False
    python unet_speed.py --gpu 0 --w $width --layers 2 --trt True
    python mm_speed.py   --gpu 0 --w $width --layers 2
} &>> "$LOG_FILE"

# Print the contents of the LOG_FILE
cat "$LOG_FILE"

# Extract and print the specific lines from the log file, then append them to the end
UNet_line=$(grep "Unt_size:" "$LOG_FILE")
UNet_linetrt=$(grep "Unt_trt:" "$LOG_FILE")
MAG_line=$(grep "MAG_size:" "$LOG_FILE")
echo -e "\n\n +---------------------------Results Summary-----------------------------+"
echo -e "$UNet_line"
echo -e "$UNet_linetrt"
echo -e "$MAG_line"

# Append the captured lines to the log file
{
    echo -e "\n\n +---------------------------Results Summary-----------------------------+"
    echo -e "$UNet_line"
    echo -e "$UNet_linetrt"
    echo -e "$MAG_line"
} >> "$LOG_FILE"