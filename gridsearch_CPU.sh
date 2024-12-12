
# Define fixed hyperparameters
DATASET="PROTEINS"
KERNELS=("RW") # You can change this as needed
HEAD=1
EPOCHS=1000
LR=0.001
LAP=True

# Output directory
OUTPUT_DIR="./Output_200_Adam"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Hyperparameters ranges
DIM_HIDDENS=(64 128)
BATCH_SIZES=(64)
DROPOUTS=(0)
FOLDS=(1 2 3 4 5 6 7 8 9 10)
LAYERS=(1 2 3 4 5 6)
HOPS=(1 2 3)
## if None WL please set WL=1, only run 1 time
ITERATIONS=(1)
GLS=(5)

# Iterate over hyperparameters
for FOLD in "${FOLDS[@]}"; do
    for DIM_HIDDEN in "${DIM_HIDDENS[@]}"; do
        for ITERATION in "${ITERATIONS[@]}"; do
            for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
                for DROPOUT in "${DROPOUTS[@]}"; do
                    for LAYER in "${LAYERS[@]}"; do
                        for HOP in "${HOPS[@]}"; do
                            echo "Running Classification_CPU.py with batch_size=$BATCH_SIZE, dropout=$DROPOUT, num_layers=$LAYER"
                            python Classification_CPU.py \
                            --dataset $DATASET \
                            --fold $FOLD \
                            --numheads $HEAD \
                            --dim_hidden $DIM_HIDDEN \
                            --num-layers $LAYER \
                            --hop $HOP \
                            --kernels ${KERNELS[*]} \
                            --fold $FOLD \
                            --dim_hidden $DIM_HIDDEN \
                            --epochs $EPOCHS \
                            --lr $LR \
                            --batch_size $BATCH_SIZE \
                            --dropout $DROPOUT \
                            --outdir $OUTPUT_DIR \
                            --wl $ITERATION 
                            done
                        done
                    done
                done
            done
        done
    done
done
