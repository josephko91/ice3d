#!/bin/bash
#PBS -A UPSU0052
#PBS -N projections-mp
#PBS -o ./out/projections-mp/output-03.log 
#PBS -e ./err/projections-mp/error-03.log
#PBS -m abe
#PBS -q main  
#PBS -l select=1:ncpus=1
#PBS -l walltime=00:10:00
#PBS -V

# Record start time
start_time=$(date +%s)  # Get current time in second

# Load conda and activate environment
module load conda
source /glade/u/apps/opt/conda/etc/profile.d/conda.sh
conda activate pyvista_pip

# set number of projections per stl file
n_proj=5
n_cores=1

# python script path 
python_script_path="/glade/u/home/joko/ice3d/scripts/python/10-projections-mp.py"

echo starting run...
echo "Taking $n_proj projections per stl file"
echo "Using $n_cores cpu cores"

# Run the Python script, passing the total number of tasks and the task index
python $python_script_path $n_proj $n_cores

# Record end time
end_time=$(date +%s)  # Get current time in seconds

# Calculate the total runtime
runtime=$((end_time - start_time))

# Print the total runtime to the output log
echo "Total runtime: $runtime seconds"
