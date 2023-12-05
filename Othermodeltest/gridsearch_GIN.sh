
# Define fixed hyperparameters
# MUTAG PTC_MR PROTEINS NCI1
DATASET="MUTAG"

# DIM_HIDDEN=64
EPOCHS=1000
LR=0.01

# Output directory
OUTPUT_DIR="./GIN"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Hyperparameters ranges
BATCH_SIZES=(32 64)

DROPOUTS=(0 0.5)
FOLDS=(1 2 3 4 5 6 7 8 9 10)
DIM_HIDDENS=(32 64)
LAYERS=(2 3 5)

# Iterate over hyperparameters
for FOLD in "${FOLDS[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        for DROPOUT in "${DROPOUTS[@]}"; do
            for LAYER in "${LAYERS[@]}"; do
                for DIM_HIDDEN in "${DIM_HIDDENS[@]}"; do
                    echo "Running othermain_GIN.py with batch_size=$BATCH_SIZE, dropout=$DROPOUT, num_layers=$LAYER"
                    python othermain_GIN.py --dataset $DATASET --fold $FOLD --num-layers $LAYER --dim_hidden $DIM_HIDDEN --epochs $EPOCHS --lr $LR --batch_size $BATCH_SIZE --dropout $DROPOUT --outdir $OUTPUT_DIR
                done
            done
        done
    done
done
