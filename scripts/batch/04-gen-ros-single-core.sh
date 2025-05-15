#!/bin/bash
#PBS -A UPSU0052
### Job name
#PBS -N stl-to-png-serial
#PBS -m abe
### queue
#PBS -q main  
### 1 node, 1 core per task
#PBS -l select=1:ncpus=1
### Time limit (4 hours)
#PBS -l walltime=04:00:00 
#PBS -V

# Record start time
start_time=$(date +%s)  # Get current time in second

# Use PBS_JOBID to dynamically set the output/error log file names after the job starts
output_log="output_${PBS_JOBID}.log"
error_log="error_${PBS_JOBID}.log"

# Echo job start message to the log files
echo "Job started at: $(date)" > $output_log
echo "Job started at: $(date)" > $error_log

echo setting up CM1 run

# Load conda and activate environment
module load conda
source /glade/u/apps/opt/conda/etc/profile.d/conda.sh
conda activate cq

# python script path 
python_script_path="/glade/u/home/joko/ice3d/scripts/python/05-gen-ros-residuals.py"

# Run the Python script and append to the logs
python $python_script_path >> $output_log 2>> $error_log

# Record end time
end_time=$(date +%s)  # Get current time in seconds

# Calculate the total runtime
runtime=$((end_time - start_time))

# Print the total runtime to the output log
echo "Total runtime: $runtime seconds" >> $output_log
