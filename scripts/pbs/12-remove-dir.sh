#!/bin/bash
#PBS -A UCLB0047
#PBS -N remove_directory
#PBS -o ./out/remove-dir/out-02.log 
#PBS -e ./err/remove-dir/err-02.log
#PBS -m abe
#PBS -q casper  
#PBS -l select=1:ncpus=1
#PBS -l walltime=04:00:00
#PBS -V

# Directory to remove (update this path as needed)
DIR_TO_REMOVE="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/projections"

# Check if the directory exists
if [ -d "$DIR_TO_REMOVE" ]; then
    echo "Removing directory: $DIR_TO_REMOVE"
    rm -rf "$DIR_TO_REMOVE"
    echo "Directory removed successfully."
else
    echo "Directory does not exist: $DIR_TO_REMOVE"
fi