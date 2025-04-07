#!/bin/bash
#SBATCH --job-name=ge_test # Job name
#SBATCH -p cpu
#SBATCH --nodes=1             # Run all processes on a single node	
#SBATCH --ntasks=1            # Number of processes
#SBATCH --cpus-per-task=1    # Number of cpus per task
#SBATCH --mem=128gb            # Total memory limit
#SBATCH --time=01:00:00       # Time limit hrs:min:sec
#SBATCH --error=/home/jko/slurm_error_logs/gen_stl_%j.err
#SBATCH --output=/home/jko/slurm_output_logs/gen_stl_log_%j.out

eval "$(conda shell.bash hook)"
conda activate 3d-modeling
pip install -e /home/jko/ice3d

python /home/jko/ice3d/scripts/python/01-gen-test.py
