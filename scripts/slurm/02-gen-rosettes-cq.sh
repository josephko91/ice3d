#!/bin/bash
#SBATCH --job-name=parallel_python_job         # Job name
#SBATCH --output=parallel_output_%A_%a.txt     # Output file (%A is job ID, %a is task ID)
#SBATCH --ntasks=N                            # Number of tasks (N parallel tasks)
#SBATCH --time=01:00:00                       # Max wall time
#SBATCH --partition=general                   # Partition to run the job on
#SBATCH --mem=4G                              # Memory per task
#SBATCH --array=0-N-1                         # Task array (0 to N-1)

# Load Python environment (adjust as needed)
module load python/3.x

# Set the input file name and output directory
INPUT_FILE="data.txt"
OUTPUT_DIR="output"
mkdir -p $OUTPUT_DIR  # Create the output directory if it doesn't exist

# Get the total number of lines in the input file
NUM_LINES=$(wc -l < $INPUT_FILE)

# Calculate the number of lines per task
LINES_PER_TASK=$((NUM_LINES / N))
START_LINE=$((SLURM_ARRAY_TASK_ID * LINES_PER_TASK + 1))
END_LINE=$((START_LINE + LINES_PER_TASK - 1))

# Handle the last task to ensure it processes all remaining lines
if [ $SLURM_ARRAY_TASK_ID -eq $((N - 1)) ]; then
    END_LINE=$NUM_LINES
fi

# Run the Python script for the chunk of data
python process_data.py "$INPUT_FILE" $START_LINE $END_LINE "$OUTPUT_DIR/output_${SLURM_ARRAY_TASK_ID}.txt"
