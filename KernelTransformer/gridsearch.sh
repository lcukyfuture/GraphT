
# Define fixed hyperparameters
DATASET="MUTAG"
KERNEL="WL_GPU" # You can change this as needed
DIM_HIDDEN=64
EPOCHS=300
LR=0.001

# Output directory
OUTPUT_DIR="./Output"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Hyperparameters ranges
BATCH_SIZES=(32 64)
DROPOUTS=(0 0.1)
FOLDS=(1 2 3 4 5 6 7 8 9 10)
LAYERS=(1 2 3 4 5 6)
HOPS=(1 2 3 4 5)
ITERATIONS=(2 3 4 5)

# Iterate over hyperparameters
for FOLD in "${FOLDS[@]}"; do
    for ITERATION in "${ITERATIONS[@]}"; do
        for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
            for DROPOUT in "${DROPOUTS[@]}"; do
                for LAYER in "${LAYERS[@]}"; do
                    for HOP in "${HOPS[@]}"; do
                        echo "Running Classification.py with batch_size=$BATCH_SIZE, dropout=$DROPOUT, num_layers=$LAYER"
                        python Classification.py --dataset $DATASET --fold $FOLD --num-layers $LAYER --hop $HOP --kernel $KERNEL --fold $FOLD --dim_hidden $DIM_HIDDEN --epochs $EPOCHS --lr $LR --batch_size $BATCH_SIZE --dropout $DROPOUT --outdir $OUTPUT_DIR --iteration $ITERATION
                    done
                done
            done
        done
    done
done
