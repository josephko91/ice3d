#!/bin/bash
#PBS -A UCLB0047
#PBS -N gen_ros_parallel
#PBS -m abe
#PBS -q main  
#PBS -o ./out/get-stl/output-07.log 
#PBS -e ./err/get-stl/error-07.log
#PBS -l select=1:ncpus=1
#PBS -l walltime=01:00:00 
#PBS -J 0-9
#PBS -V

# Record start time
start_time=$(date +%s)  # Get current time in second

echo setting up CM1 run ${PBS_ARRAY_INDEX}

# Load conda and activate environment
module load conda
source /glade/u/apps/opt/conda/etc/profile.d/conda.sh
conda activate cq

# Number of tasks
num_tasks=10  # Here, we want 10 tasks

# # Get the task index (PBS_ARRAYID provides the index for each job in the array)
# task_index=${PBS_ARRAY_INDEX}

# echo $task_index

# python script path 
python_script_path="/glade/u/home/joko/ice3d/scripts/python/02-gen-rosettes-cq.py"

# Run the Python script, passing the total number of tasks and the task index
python $python_script_path $num_tasks ${PBS_ARRAY_INDEX}


# Record end time
end_time=$(date +%s)  # Get current time in seconds

# Calculate the total runtime
runtime=$((end_time - start_time))

# Print the total runtime to the output log
echo "Total runtime: $runtime seconds"
