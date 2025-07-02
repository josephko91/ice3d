#!/bin/bash

SAMPLEDIR="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/projections/phips/6"
NUMFILES=10000

if [[ -z "$SAMPLEDIR" || ! -d "$SAMPLEDIR" ]]; then
    echo "Usage: $0 /path/to/sample_dir [num_files]"
    exit 1
fi

echo "Sampling $NUMFILES files from: $SAMPLEDIR"
cd "$SAMPLEDIR" || exit 2

echo "Timing compression of $NUMFILES files..."
START=$(date +%s)
find . -type f | head -n "$NUMFILES" | tar --use-compress-program="zstd -T0 --fast" -cf /dev/null -T -
END=$(date +%s)

ELAPSED=$((END - START))
echo "Elapsed time for $NUMFILES files: $ELAPSED seconds"

# Optional: Estimate total time if you know the total number of files
TOTAL_FILES=$(find . -type f | wc -l)
if [[ "$TOTAL_FILES" -gt "$NUMFILES" ]]; then
    ESTIMATED_TOTAL=$((ELAPSED * TOTAL_FILES / NUMFILES))
    echo "Estimated time for all $TOTAL_FILES files: $ESTIMATED_TOTAL seconds (~$((ESTIMATED_TOTAL/60)) minutes)"
fi