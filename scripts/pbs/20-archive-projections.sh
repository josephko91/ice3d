#!/bin/bash
#PBS -A UCLB0047
#PBS -m abe
#PBS -q casper  
#PBS -o ./out/archive-projections/output-04.log 
#PBS -N archive_directory
#PBS -l select=1:ncpus=18
#PBS -l walltime=08:00:00
#PBS -j oe
#PBS -J 1-22

SRC_BASE="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/projections"
DEST_BASE="/glade/derecho/scratch/joko/synth-ros/params_200_50_final_archive/projections"
LEAF_LIST="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/projections/leaf_dirs.txt"
CHUNK_SIZE=10000
mkdir -p "$DEST_BASE"

# Get the directory for this array task
DIR=$(sed -n "${PBS_ARRAY_INDEX}p" "$LEAF_LIST")
if [ -z "$DIR" ]; then
    echo "No directory found for PBS_ARRAY_INDEX=$PBS_ARRAY_INDEX"
    exit 1
fi

rel_path="${DIR#$SRC_BASE/}"

echo "Processing $DIR in chunks of $CHUNK_SIZE files..."

# List all files, split into chunks, and archive each chunk
tmp_prefix="$DEST_BASE/tmp_filelist_${rel_path//\//_}_"
find "$DIR" -type f -printf '%P\n' | split -l $CHUNK_SIZE - "$tmp_prefix"

chunk=0
for filelist in "${tmp_prefix}"*; do
    # Only process if the filelist exists and is not empty
    [ -s "$filelist" ] || continue
    archive_name="${rel_path//\//_}_chunk${chunk}.tar.zst"
    archive_path="$DEST_BASE/$archive_name"
    echo "  Archiving chunk $chunk to $archive_path"
    tar --use-compress-program="zstd -T0 --fast" -cf "$archive_path" -C "$DIR" --files-from="$filelist"
    if [ $? -eq 0 ]; then
        echo "  Archive created successfully: $archive_path"
    else
        echo "  Error archiving chunk $chunk in $DIR"
        rm -f "$filelist"
        exit 1
    fi
    rm -f "$filelist"
    chunk=$((chunk + 1))
done