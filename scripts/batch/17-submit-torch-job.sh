#!/bin/bash -l
#PBS -N fine-tune
#PBS -A UCLB0041
#PBS -o ./out/torch-training/out-03.log 
#PBS -e ./err/torch-training/err-03.log
#PBS -l select=1:ncpus=64:ngpus=4:mem=500GB
#PBS -l gpu_type=a100
#PBS -l walltime=08:00:00
#PBS -m abe
#PBS -q casper
#PBS -V

### Provide CUDA runtime libraries
module load cuda
module load conda 
conda activate torch

# python script path 
python_script_path="/glade/u/home/joko/ice3d/scripts/python/12-train-torch-models.py"

### Measure execution time
start_time=$(date +%s)

### Run program
# Run the Python script and append to the logs
python $python_script_path

### Calculate and print execution time
end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Total execution time: ${execution_time} seconds"