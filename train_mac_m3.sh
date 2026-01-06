#!/bin/bash

# Diffusion Training Launch Script for Mac M3 Ultra
# Usage: ./train_mac_m3.sh

# ============================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================

EXPERIMENT_NAME="celebA_diffusion_m3_ultra"
PATH_TO_DATA="/Users/kkpatro/Desktop/Diffusion/celeb_hq_128"  # CHANGE THIS
WORKING_DIR="/Users/kkpatro/Desktop/ddpm/working"
GENERATED_DIR="/Users/kkpatro/Desktop/ddpm/generated"

# ============================================
# TRAINING HYPERPARAMETERS
# ============================================

# IMPORTANT: Reduced settings for MPS compatibility
BATCH_SIZE=16                    # Reduced from 32
IMG_SIZE=128
STARTING_CHANNELS=96             # Reduced from 128 to avoid MPS overflow
NUM_TRAINING_STEPS=100000
EVALUATION_INTERVAL=5000
LEARNING_RATE=0.0001
WARMUP_STEPS=5000
NUM_WORKERS=4

# ============================================
# OPTIONAL FLAGS
# ============================================

# Uncomment to use Weights & Biases
# USE_WANDB="--use_wandb"
USE_WANDB=""

# Uncomment to resume from checkpoint
# RESUME_CHECKPOINT="--resume_from_checkpoint checkpoint_10000"
RESUME_CHECKPOINT=""

# ============================================
# CREATE DIRECTORIES
# ============================================

mkdir -p $WORKING_DIR
mkdir -p $GENERATED_DIR

# ============================================
# CHECK PYTORCH MPS AVAILABILITY
# ============================================

echo "Checking PyTorch MPS availability..."
python3 -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}'); print(f'PyTorch Version: {torch.__version__}')"

if [ $? -ne 0 ]; then
    echo "Error: PyTorch not found. Please install PyTorch first:"
    echo "pip install torch torchvision torchaudio"
    exit 1
fi

# ============================================
# LAUNCH TRAINING
# ============================================

echo ""
echo "=========================================="
echo "Starting Diffusion Training on Mac M3 Ultra"
echo "=========================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Image Size: $IMG_SIZE x $IMG_SIZE"
echo "Batch Size: $BATCH_SIZE"
echo "Training Steps: $NUM_TRAINING_STEPS"
echo "=========================================="
echo ""

python3 diffusion_mac_m3.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --path_to_data "$PATH_TO_DATA" \
    --working_directory "$WORKING_DIR" \
    --generated_directory "$GENERATED_DIR" \
    --batch_size $BATCH_SIZE \
    --img_size $IMG_SIZE \
    --starting_channels $STARTING_CHANNELS \
    --num_training_steps $NUM_TRAINING_STEPS \
    --evaluation_interval $EVALUATION_INTERVAL \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --num_workers $NUM_WORKERS \
    --loss_fn mse \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --num_keep_checkpoints 2 \
    --num_generations 5 \
    --plot_freq_interval 100 \
    $USE_WANDB \
    $RESUME_CHECKPOINT

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Checkpoints saved to: $WORKING_DIR/$EXPERIMENT_NAME"
echo "Generated images saved to: $GENERATED_DIR"
echo "=========================================="