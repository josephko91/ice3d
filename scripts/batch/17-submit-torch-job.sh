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

module load cuda
module load conda 
conda activate torch

python_script_path="/glade/u/home/joko/ice3d/scripts/python/12-train-torch-models.py"

# Set your arguments here
MODEL="resnet18_regression"
DATA_TYPE="single_view_h5"
HDF_FILE="/glade/u/home/joko/ice3d/data/mydata.h5"
TARGETS="rho_eff,sa_eff"
INPUT_CHANNELS=2
BATCH_SIZE=64
LR=1e-4
MAX_EPOCHS=20
LOG_DIR="./lightning_logs"
TB_LOG_NAME="tb"
CSV_LOG_NAME="csv"

start_time=$(date +%s)

python $python_script_path \
    --model $MODEL \
    --data_type $DATA_TYPE \
    --hdf_file $HDF_FILE \
    --targets $TARGETS \
    --input_channels $INPUT_CHANNELS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --max_epochs $MAX_EPOCHS \
    --log_dir $LOG_DIR \
    --tb_log_name $TB_LOG_NAME \
    --csv_log_name $CSV_LOG_NAME

end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Total execution time: ${execution_time} seconds"