#!/bin/bash
#PBS -A UCLB0047
#PBS -N copy_directory
#PBS -o ./out/copy-dir/out-01.log 
#PBS -e ./err/copy-dir/err-01.log
#PBS -m abe
#PBS -q main  
#PBS -l select=1:ncpus=1
#PBS -l walltime=01:00:00
#PBS -V

# Source and destination directories (update these paths as needed)
SRC_DIR="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/projections/default/4"
DEST_DIR="/glade/work/joko/projections-backup/default"

# Check if the source directory exists
if [ -d "$SRC_DIR" ]; then
    echo "Copying directory from $SRC_DIR to $DEST_DIR"
    rsync -a --delete --partial "$SRC_DIR" "$DEST_DIR/"
    echo "Directory copied successfully."
else
    echo "Source directory does not exist: $SRC_DIR"
fi