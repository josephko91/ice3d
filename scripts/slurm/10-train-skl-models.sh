#!/bin/bash
#PBS -A UPSU0052
#PBS -N train_skl
#PBS -o ./out/skl-train/output-01.log 
#PBS -e ./err/skl-train/error-01.log
#PBS -m abe
#PBS -q main  
#PBS -l select=1:ncpus=100
#PBS -l walltime=01:00:00
#PBS -V

# Record start time
start_time=$(date +%s)  # Get current time in second

# Load conda and activate environment
module load conda
source /glade/u/apps/opt/conda/etc/profile.d/conda.sh
conda activate pyvista_pip

# set parameters for python script
n_cores=100
save_dir="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/projections"
data_dir="/glade/derecho/scratch/joko/synth-ros/params_200_50-debug-20250316/tabular-data"
python_script_path="/glade/u/home/joko/ice3d/scripts/python/13-train-skl-models.py"
view="default"

echo starting run...
echo "Using $n_cores cpu cores"

# Run the Python script, passing the total number of tasks and the task index
python $python_script_path $n_cores $save_dir $data_dir $view

# Record end time
end_time=$(date +%s)  # Get current time in seconds

# Calculate the total runtime
runtime=$((end_time - start_time))

# Print the total runtime to the output log
echo "Total runtime: $runtime seconds"
