#!/bin/bash
#PBS -A UPSU0052
### Job name
#PBS -N projections_two_view_parallel
#PBS -o ./out/n200-t100-cpus10/output-01.log 
#PBS -e ./err/n200-t100-cpus10/error-01.log
#PBS -m abe
### queue
#PBS -q main  
### select # nodes and cores
#PBS -l select=1:ncpus=10
### Time limit
#PBS -l walltime=01:00:00 
### Job array (N tasks, from idx 0-(N-1))
#PBS -J 0-99%100
#PBS -V

# Record start time
start_time=$(date +%s)  # Get current time in second

echo setting up CM1 run ${PBS_ARRAY_INDEX}

# Load conda and activate environment
module load conda
source /glade/u/apps/opt/conda/etc/profile.d/conda.sh
conda activate pyvista_pip

# Number of tasks
num_tasks=100

# python script path 
python_script_path="/glade/u/home/joko/ice3d/scripts/python/08-stl-to-png.py"

# Run the Python script, passing the total number of tasks and the task index
python $python_script_path $num_tasks ${PBS_ARRAY_INDEX}

# Record end time
end_time=$(date +%s)  # Get current time in seconds

# Calculate the total runtime
runtime=$((end_time - start_time))

# Print the total runtime to the output log
echo "Total runtime: $runtime seconds"
