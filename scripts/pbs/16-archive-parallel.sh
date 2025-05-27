#!/bin/bash
#PBS -A UCLB0047
#PBS -m abe
#PBS -q casper
#PBS -o ./out/archive/output-10.log
#PBS -e ./err/archive/error-10.log
#PBS -N archive_directory_fast
#PBS -l select=1:ncpus=62:mem=100G
#PBS -l walltime=01:00:00

# Start timer
START_TIME=$(date +%s)

# Variables
SOURCE_DIR="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/subset_n1000"
OUTPUT_FILE="/glade/derecho/scratch/joko/test.tar.gz"
PIGZ_THREADS=62

# Move to source parent directory
cd "$(dirname "$SOURCE_DIR")" || exit 1

# Archive using tar with pigz (parallel gzip)
echo "Archiving $SOURCE_DIR to $OUTPUT_FILE with pigz using $PIGZ_THREADS threads"
tar -cf "$OUTPUT_FILE" \
    --checkpoint=20000 \
    --checkpoint-action=echo='Archived ~10MB...' \
    -I "pigz -p $PIGZ_THREADS" \
    "$(basename "$SOURCE_DIR")"

# Check success
if [ $? -eq 0 ]; then
    echo "Archive created successfully: $OUTPUT_FILE"
else
    echo "Archiving failed."
    exit 1
fi

# End timer
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

# Format and display time
echo "Total execution time: $ELAPSED_TIME seconds ($(date -ud "@$ELAPSED_TIME" +'%Hh %Mm %Ss'))"