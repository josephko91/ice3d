#!/bin/bash
#PBS -N ml-regression-test
#PBS -A UCLB0041
#PBS -o ./out/torch-training/out-24.log 
#PBS -e ./err/torch-training/err-24.log
#PBS -l select=1:ncpus=18:ngpus=2:mem=100GB
#PBS -l gpu_type=v100
#PBS -l walltime=00:30:00
#PBS -m abe
#PBS -q casper
#PBS -V

module load cuda
module load conda 
conda activate torch

python_script_path="/glade/u/home/joko/ice3d/scripts/python/12-train-torch-models.py"

# Set your arguments here
MODEL="mlp_regression"
DATA_TYPE="tabular"  # Options: tabular, single_view_h5, stereo_view_h5
FEATURE_NAMES="aspect_ratio,aspect_ratio_elip,extreme_pts,contour_area,contour_perimeter,area_ratio,complexity,circularity"  # <-- Set your feature columns here
# Tabular data
TABULAR_FILE="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/tabular-data-v2/ros-tabular-data.parquet"
# Single view data
HDF_FILE="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/imgs-ml-ready/default.h5"
# Stereo view data
HDF_FILE_LEFT="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/imgs-ml-ready/default.h5"
HDF_FILE_RIGHT="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/imgs-ml-ready/phips.h5"
TARGETS="rho_eff,sa_eff"
# TARGETS="n_arms"
INPUT_CHANNELS=1
BATCH_SIZE=128
LR=1e-3
MAX_EPOCHS=2
SUBSET_SIZE=0.1
SEED=666
NUM_WORKERS=18
NUM_GPUS=2
PREFETCH_FACTOR=16
TASK_TYPE="regression"  # "regression" or "classification"
LOG_DIR="/glade/u/home/joko/ice3d/models/lightning_logs"
TB_LOG_NAME="mlp-regression-subset-tb"
CSV_LOG_NAME="mlp-regression-subset-csv"
# Specify the class mapping JSON file (set to empty string if not used)
CLASS_TO_IDX_JSON="/glade/u/home/joko/ice3d/data/class_to_idx.json"
# start timer
start_time=$(date +%s)

# Build the dataset-specific arguments
DATA_ARGS=""
if [ "$DATA_TYPE" = "tabular" ]; then
    DATA_ARGS="--tabular_file $TABULAR_FILE"
elif [ "$DATA_TYPE" = "single_view_h5" ]; then
    DATA_ARGS="--hdf_file $HDF_FILE"
elif [ "$DATA_TYPE" = "stereo_view_h5" ]; then
    DATA_ARGS="--hdf_file_left $HDF_FILE_LEFT --hdf_file_right $HDF_FILE_RIGHT"
fi

# Optional index files (set to empty string if not used)
TRAIN_IDX=""
VAL_IDX=""
TEST_IDX=""

# Add to DATA_ARGS if provided
if [ -n "$TRAIN_IDX" ]; then
    DATA_ARGS="$DATA_ARGS --train_idx $TRAIN_IDX"
fi
if [ -n "$VAL_IDX" ]; then
    DATA_ARGS="$DATA_ARGS --val_idx $VAL_IDX"
fi
if [ -n "$TEST_IDX" ]; then
    DATA_ARGS="$DATA_ARGS --test_idx $TEST_IDX"
fi

python $python_script_path \
    --model $MODEL \
    --data_type $DATA_TYPE \
    $DATA_ARGS \
    --targets $TARGETS \
    --input_channels $INPUT_CHANNELS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --max_epochs $MAX_EPOCHS \
    --subset_size $SUBSET_SIZE \
    --seed $SEED \
    --num_workers $NUM_WORKERS \
    --prefetch_factor $PREFETCH_FACTOR \
    --task_type $TASK_TYPE \
    --log_dir $LOG_DIR \
    --tb_log_name $TB_LOG_NAME \
    --csv_log_name $CSV_LOG_NAME \
    --num_gpus $NUM_GPUS \
    --class_to_idx $CLASS_TO_IDX_JSON \
    --feature_names $FEATURE_NAMES

end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Total execution time: ${execution_time} seconds"