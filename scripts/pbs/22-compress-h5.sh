#!/bin/bash
#PBS -A UCLB0047
#PBS -m abe
#PBS -q casper  
#PBS -o ./out/compress-h5/output-08.log 
#PBS -N compress_phips_sorted
#PBS -l select=1:ncpus=1
#PBS -l walltime=23:00:00
#PBS -j oe

# use h5repack to compress HDF5 files
INPUT_H5="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/imgs-ml-ready/sorted/phips_sorted.h5"
OUTPUT_H5="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/imgs-ml-ready/sorted/phips_sorted_compressed.h5"

echo "Starting h5repack at $(date)"
START_TIME=$(date +%s)

h5repack -v -f GZIP=9 "$INPUT_H5" "$OUTPUT_H5"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "h5repack finished at $(date)"
echo "Total elapsed time: ${ELAPSED}"