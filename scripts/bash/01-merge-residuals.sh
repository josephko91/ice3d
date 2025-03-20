#!/bin/bash

# Get the source and destination directories from arguments
SOURCE_DIR="/glade/derecho/scratch/joko/synth-ros/params_200_50-debug-20250316"
DEST_DIR="/glade/derecho/scratch/joko/synth-ros/params_200_50-merged-20250320"

# Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist."
    exit 1
fi

# Create the destination directory if it doesn't exist
if [ ! -d "$DEST_DIR" ]; then
    echo "Destination directory '$DEST_DIR' does not exist. Creating it..."
    mkdir -p "$DEST_DIR"
fi

# Copy the source directory to the destination
cp -r "$SOURCE_DIR" "$DEST_DIR"

# Check if the copy was successful
if [ $? -eq 0 ]; then
    echo "Directory '$SOURCE_DIR' successfully copied to '$DEST_DIR'."
else
    echo "Error: Failed to copy the directory."
    exit 1
fi

# Define the paths for 'stl' and 'residuals/stl' in the destination
STL_DIR="$DEST_DIR/stl"
RESIDUALS_STL_DIR="$DEST_DIR/residuals/stl"

# Check if the 'stl' and 'residuals/stl' directories exist
if [ ! -d "$STL_DIR" ] || [ ! -d "$RESIDUALS_STL_DIR" ]; then
    echo "Error: One or both directories 'stl' or 'residuals/stl' do not exist in the destination."
    exit 1
fi

# Move files from 'residuals/stl' to 'stl', preserving the structure
find "$RESIDUALS_STL_DIR" -type f | while read file; do
    # Get the relative path of the file from 'residuals/stl'
    relative_path="${file#$RESIDUALS_STL_DIR/}"
    
    # Construct the destination path in 'stl'
    destination_file="$STL_DIR/$relative_path"
    
    # Ensure the destination directory exists
    destination_dir=$(dirname "$destination_file")
    if [ ! -d "$destination_dir" ]; then
        mkdir -p "$destination_dir"
    fi
    
    # Check if the file already exists in the destination, and only move if it doesn't exist
    if [ ! -f "$destination_file" ]; then
        mv "$file" "$destination_file"
        
        # Check if the move was successful
        if [ $? -eq 0 ]; then
            echo "Moved '$file' to '$destination_file'."
        else
            echo "Error: Failed to move '$file'."
        fi
    else
        echo "Skipping '$file' as it already exists in '$destination_file'."
    fi
done

echo "All files from 'residuals/stl' have been moved to 'stl' successfully (no overwrites)."

# Define the paths for 'data' and 'residuals/data' in the destination
DATA_DIR="$DEST_DIR/data"
RESIDUALS_DATA_DIR="$DEST_DIR/residuals/data"

# Check if the 'data' and 'residuals/data' directories exist
if [ ! -d "$DATA_DIR" ] || [ ! -d "$RESIDUALS_DATA_DIR" ]; then
    echo "Error: One or both directories 'data' or 'residuals/data' do not exist in the destination."
    exit 1
fi

# Move files from 'residuals/data' to 'data', preserving the structure
find "$RESIDUALS_DATA_DIR" -type f | while read file; do
    # Get the relative path of the file from 'residuals/data'
    relative_path="${file#$RESIDUALS_DATA_DIR/}"
    
    # Construct the destination path in 'data'
    destination_file="$DATA_DIR/$relative_path"
    
    # Ensure the destination directory exists
    destination_dir=$(dirname "$destination_file")
    if [ ! -d "$destination_dir" ]; then
        mkdir -p "$destination_dir"
    fi
    
    # Check if the file already exists in the destination, and only move if it doesn't exist
    if [ ! -f "$destination_file" ]; then
        mv "$file" "$destination_file"
        
        # Check if the move was successful
        if [ $? -eq 0 ]; then
            echo "Moved '$file' to '$destination_file'."
        else
            echo "Error: Failed to move '$file'."
        fi
    else
        echo "Skipping '$file' as it already exists in '$destination_file'."
    fi
done

echo "All files from 'residuals/data' have been moved to 'data' successfully (no overwrites)."