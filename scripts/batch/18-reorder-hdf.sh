#!/bin/bash
#PBS -A UCLB0041
#PBS -m abe
#PBS -q casper
#PBS -o ./out/reorder-hdf/output-12.log
#PBS -e ./err/reorder-hdf/error-12.log
#PBS -N reorder_2ds_shuffled
#PBS -l select=1:ncpus=1:mem=500G
#PBS -l walltime=04:00:00

# Start timer
START_TIME=$(date +%s)

# Load conda and activate environment
module load conda
conda activate npl

# Set number of cpus
# NUM_CPUS=10

# Specify paths
PYTHON_SCRIPT="/glade/u/home/joko/ice3d/scripts/python/18-reorder-hdf-serial.py"
FILENAME_ORDER="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/tabular-data-v2/filenames_shuffled_2ds.txt"
HDF_FILE="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/imgs-ml-ready/2ds.h5"
SAVE_DIR="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/imgs-ml-ready/shuffled"
CHUNK_SIZE=512

python -u $PYTHON_SCRIPT \
    --hdf_file $HDF_FILE \
    --filename_order $FILENAME_ORDER \
    --save_dir $SAVE_DIR \
    --chunk_size $CHUNK_SIZE \
    # --num_cpus $NUM_CPUS

# End timer
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

# Format and display time
echo "Total execution time: $ELAPSED_TIME seconds ($(date -ud "@$ELAPSED_TIME" +'%Hh %Mm %Ss'))"