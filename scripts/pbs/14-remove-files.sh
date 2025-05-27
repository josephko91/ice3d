#!/bin/bash
#PBS -A UCLB0047
#PBS -N remove_files_dry_run
#PBS -o ./out/remove-files/out-03.log 
#PBS -e ./err/remove-files/err-03.log
#PBS -m abe
#PBS -q main  
#PBS -l select=1:ncpus=1:mem=4GB
#PBS -l walltime=08:00:00
#PBS -V

# Define base path and subdirectories
BASE_PATH="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/projections"
SUBDIRS=("default" "2ds" "phips")

# Create a temporary empty directory
EMPTY_DIR="/tmp/empty_dir"
mkdir -p "$EMPTY_DIR"

# Loop through subdirectories and remove files in the "4" subfolder
for SUBDIR in "${SUBDIRS[@]}"; do
    TARGET_DIR="$BASE_PATH/$SUBDIR/4"
    if [ -d "$TARGET_DIR" ]; then
        echo "Removing files in $TARGET_DIR using rsync"
        rsync -a --delete "$EMPTY_DIR/" "$TARGET_DIR/"
        echo "All files removed in subdirectory: $SUBDIR"
    else
        echo "Directory does not exist: $TARGET_DIR"
    fi
done

# Clean up the temporary empty directory
rmdir "$EMPTY_DIR"