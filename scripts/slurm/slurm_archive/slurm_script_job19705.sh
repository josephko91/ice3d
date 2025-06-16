#!/bin/bash
#SBATCH --job-name=resnet-cls-stereo-2ds-b128-e50
#SBATCH --account=ai2es_premium
#SBATCH --output=./out/torch-training/out-19.log
#SBATCH --error=./err/torch-training/err-19.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --mem=200GB
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jk4730@columbia.edu

# Save a copy of this script to a logs folder using the SLURM job ID
SCRIPT_NAME=$(basename "$0")
ARCHIVE_DIR="./slurm_archive"
ARCHIVE_NAME="${SCRIPT_NAME%.sh}_job${SLURM_JOB_ID}.sh"
mkdir -p "$ARCHIVE_DIR"
cp "$0" "$ARCHIVE_DIR/$ARCHIVE_NAME"

eval "$(conda shell.bash hook)"
conda activate torch

python_script_path="/home/jko/ice3d/scripts/python/12-train-torch-models.py"
# Set your arguments here
MODEL="resnet18_classification"
DATA_TYPE="stereo_view_h5"  # Options: tabular, single_view_h5, stereo_view_h5
FEATURE_NAMES="aspect_ratio,aspect_ratio_elip,extreme_pts,contour_area,contour_perimeter,area_ratio,complexity,circularity"
TABULAR_FILE="/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/tabular-data-v2/ros-tabular-data.parquet"
HDF_FILE="/home/jko/synth-ros-data/imgs-ml-ready/shuffled_small/default_shuffled_small.h5"
HDF_FILE_LEFT="/home/jko/synth-ros-data/imgs-ml-ready/shuffled_small/default_shuffled_small.h5"
HDF_FILE_RIGHT="/home/jko/synth-ros-data/imgs-ml-ready/shuffled_small/2ds_shuffled_small.h5"
TARGETS="n_arms"
INPUT_CHANNELS=2
BATCH_SIZE=128
LR=1e-3
MAX_EPOCHS=50
SUBSET_SIZE=1.0
SEED=666
NUM_WORKERS=32
NUM_GPUS=1
PREFETCH_FACTOR=32
TASK_TYPE="classification"
LOG_DIR="/home/jko/ice3d/models/lightning_logs"
TB_LOG_NAME="resnet18-classification-stereo-2ds-subset-700k-tb"
CSV_LOG_NAME="resnet18-classification-stereo-2ds-subset-700k-csv"
CLASS_TO_IDX_JSON="/home/jko/ice3d/data/class_to_idx.json"

start_time=$(date +%s)

DATA_ARGS=""
if [ "$DATA_TYPE" = "tabular" ]; then
    DATA_ARGS="--tabular_file $TABULAR_FILE"
elif [ "$DATA_TYPE" = "single_view_h5" ]; then
    DATA_ARGS="--hdf_file $HDF_FILE"
elif [ "$DATA_TYPE" = "stereo_view_h5" ]; then
    DATA_ARGS="--hdf_file_left $HDF_FILE_LEFT --hdf_file_right $HDF_FILE_RIGHT"
fi

TRAIN_IDX="/home/jko/synth-ros-data/idx/idx-train-sequential-subset-700k.txt"
VAL_IDX="/home/jko/synth-ros-data/idx/idx-val-sequential-subset-700k.txt"
TEST_IDX="/home/jko/synth-ros-data/idx/idx-test-sequential-subset-700k.txt"

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
    --class_to_idx_json $CLASS_TO_IDX_JSON \
    --feature_names $FEATURE_NAMES

end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Total execution time: ${execution_time} seconds"