#!/bin/bash

# This script creates a list of all leaf directories in the specified base directory as saves as a text file in specified directory
BASE_DIR="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/projections"
OUT_DIR="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/projections"

mkdir -p "$OUT_DIR"

# Generate list of leaf directories (i.e., terminal directories with no subdirectories)
LEAF_LIST="$OUT_DIR/leaf_dirs.txt"
if [ ! -f "$LEAF_LIST" ]; then 
    find "$BASE_DIR" -type d -links 2 | sort > "$LEAF_LIST"
fi