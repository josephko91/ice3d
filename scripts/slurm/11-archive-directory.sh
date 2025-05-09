#!/bin/bash
#PBS -A UCLB0047
#PBS -m abe
#PBS -q main  
#PBS -o ./out/archive/output-04.log 
#PBS -e ./err/archive/error-04.log
#PBS -N archive_directory
#PBS -l select=1:ncpus=64
#PBS -l walltime=08:00:00
#PBS -j oe

# Define the directory to archive and the output file
SOURCE_DIR="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403"
OUTPUT_FILE="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403-archive-20250417.tar.zst"

# Create the tar archive with zstd compression
echo "Archiving directory: $SOURCE_DIR"
tar --use-compress-program="zstd -T64 --fast" -cf "$OUTPUT_FILE" -C "$(dirname "$SOURCE_DIR")" "$(basename "$SOURCE_DIR")"

if [ $? -eq 0 ]; then
    echo "Archive created successfully: $OUTPUT_FILE"
else
    echo "Error occurred during archiving."
    exit 1
fi