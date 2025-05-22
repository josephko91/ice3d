#!/bin/bash
#PBS -A UCLB0041
#PBS -m abe
#PBS -q casper
#PBS -o ./out/subset-hdf/output-03.log
#PBS -e ./err/subset-hdf/error-03.log
#PBS -N subset_phips
#PBS -l select=1:ncpus=1:mem=500G
#PBS -l walltime=01:00:00

module load conda 
conda activate torch

PYTHON_SCRIPT="/glade/u/home/joko/ice3d/scripts/python/19-subset-hdf.py"
INPUT_HDF="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/imgs-ml-ready/shuffled/phips_shuffled.h5"
OUTPUT_HDF="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/imgs-ml-ready/shuffled_small/phips_shuffled_small.h5"
SUBSET_SIZE=700_000
CHUNK_SIZE=512

python $PYTHON_SCRIPT \
    --input_hdf $INPUT_HDF \
    --output_hdf $OUTPUT_HDF \
    --subset_size $SUBSET_SIZE \
    --chunk_size $CHUNK_SIZE \