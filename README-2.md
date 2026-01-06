# Denoising Diffusion Probabilistic Model (DDPM)

A PyTorch implementation of Denoising Diffusion Probabilistic Models for high-quality image generation. This implementation is optimized for Apple Silicon (M3 Ultra) using Metal Performance Shaders (MPS) and includes a flexible UNet architecture with self-attention mechanisms.

## Overview

This project implements the DDPM algorithm introduced in ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239) by Ho et al. The model learns to generate images by iteratively denoising random Gaussian noise through a learned reverse diffusion process.

### Key Features

- **Apple Silicon Optimization**: Native MPS support for M1/M2/M3 chips
- **Flexible Architecture**: Configurable UNet with residual blocks and transformer attention
- **Multiple Loss Functions**: Support for MSE, MAE, and Huber loss
- **Training Monitoring**: Integrated visualization of the denoising process during training
- **Checkpoint Management**: Automatic saving and cleanup of training checkpoints
- **Weights & Biases Integration**: Optional experiment tracking with W&B

## Architecture

The model consists of three main components:

1. **DDPM Sampler**: Implements the forward and reverse diffusion processes using a linear beta schedule
2. **UNet Backbone**: Encoder-decoder architecture with skip connections
3. **Time Embeddings**: Sinusoidal positional embeddings for diffusion timestep conditioning

### Model Components

- **Residual Blocks**: Conv2d layers with GroupNorm and time embedding injection
- **Self-Attention**: Multi-head attention for capturing long-range dependencies
- **Encoder-Decoder**: Progressive downsampling and upsampling with skip connections

## Requirements

```bash
torch>=2.0.0
torchvision
numpy
matplotlib
Pillow
tqdm
transformers
```

### Optional
```bash
wandb  # For experiment tracking
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ddpm-implementation.git
cd ddpm-implementation

# Install dependencies
pip install torch torchvision numpy matplotlib Pillow tqdm transformers

# Optional: Install W&B for experiment tracking
pip install wandb
```

## Project Structure

```
ddpm-implementation/
├── diffusion_mac_m3.py          # Main training script
├── train_mac_m3.sh              # Convenient launch script for Mac M3
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── generated/                   # Generated samples during training
│   └── step_*.png              # Denoising visualizations
├── working/                     # Training checkpoints and logs
│   └── experiment_name/
│       └── checkpoint_*/
│           └── model.pt        # Model checkpoint
└── data/                        # Your dataset directory
    └── celeb_hq_128/           # Example: CelebA 128x128
        ├── 000001.jpg
        ├── 000002.jpg
        └── ...
```

## Dataset Preparation

This implementation is designed for the CelebA dataset, but can work with any image dataset:

1. Download your dataset (e.g. I have used celeb HQ data set of 30,000 images )
2. Organize images in a single directory
3. Supported formats: PNG, JPG, JPEG, BMP, GIF

## Usage

### Quick Start with Training Script

A convenient bash script (`train_mac_m3.sh`) is provided for easy training on Mac M3:

```bash
# Make the script executable
chmod +x train_mac_m3.sh

# Edit the paths in the script, then run
./train_mac_m3.sh
```

The script automatically:
- Checks PyTorch MPS availability
- Creates necessary directories
- Launches training with optimized parameters
- Displays training progress

**Edit these paths in `train_mac_m3.sh`:**
```bash
PATH_TO_DATA="/path/to/your/dataset"
WORKING_DIR="/path/to/checkpoints"
GENERATED_DIR="/path/to/generated/samples"
```

### Recommended Configuration (Mac M3 Ultra)

The following configuration was used for training on Mac M3 Ultra with 128x128 images:

```bash
python3 diffusion_mac_m3.py \
  --experiment_name "celebA_diffusion_m3_ultra" \
  --path_to_data "./celeb_hq_128" \
  --working_directory "./working" \
  --generated_directory "./generated" \
  --batch_size 16 \
  --img_size 128 \
  --starting_channels 96 \
  --num_training_steps 100000 \
  --evaluation_interval 5000 \
  --learning_rate 0.0001 \
  --warmup_steps 5000 \
  --num_workers 4 \
  --loss_fn mse \
  --max_grad_norm 1.0 \
  --weight_decay 0.0001 \
  --num_keep_checkpoints 2 \
  --num_generations 5 \
  --plot_freq_interval 100
```

**Key Settings for M3 Ultra:**
- **Batch Size**: 16 (reduced for MPS memory management)
- **Starting Channels**: 96 (reduced from 128 to avoid MPS overflow)
- **Image Size**: 128x128 (balance between quality and speed)
- **Workers**: 4 (optimized for Mac architecture)

### Advanced Configuration

For different hardware or requirements:

```bash
python diffusion_mac_m3.py \
  --experiment_name "ddpm_experiment" \
  --path_to_data "./data/celeba" \
  --working_directory "./checkpoints" \
  --generated_directory "./samples" \
  --num_diffusion_timesteps 1000 \
  --num_training_steps 200000 \
  --batch_size 32 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --warmup_steps 5000 \
  --evaluation_interval 5000 \
  --img_size 128 \
  --starting_channels 128 \
  --loss_fn "mse" \
  --use_wandb
```

### Resume Training

```bash
# Using the training script
# Uncomment in train_mac_m3.sh:
RESUME_CHECKPOINT="--resume_from_checkpoint checkpoint_50000"

# Or directly:
python diffusion_mac_m3.py \
  --experiment_name "ddpm_celeba" \
  --resume_from_checkpoint "checkpoint_50000" \
  [other arguments...]
```

## Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--experiment_name` | Name of the training run |
| `--path_to_data` | Path to image dataset directory |
| `--working_directory` | Directory for storing checkpoints and logs |
| `--generated_directory` | Directory for storing generated samples |

### Model Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--img_size` | 128 | Image width and height |
| `--starting_channels` | 128 | Number of channels in first convolution |
| `--num_diffusion_timesteps` | 1000 | Number of diffusion timesteps |

### Training Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_training_steps` | 150000 | Total training steps |
| `--batch_size` | 32 | Batch size per step |
| `--gradient_accumulation_steps` | 1 | Gradient accumulation steps |
| `--learning_rate` | 1e-4 | Maximum learning rate |
| `--warmup_steps` | 5000 | Learning rate warmup steps |
| `--weight_decay` | 1e-4 | Weight decay for AdamW |
| `--max_grad_norm` | 1.0 | Maximum gradient norm for clipping |
| `--loss_fn` | mse | Loss function (mse/mae/huber) |

### Evaluation & Logging

| Argument | Default | Description |
|----------|---------|-------------|
| `--evaluation_interval` | 5000 | Steps between evaluations |
| `--plot_freq_interval` | 100 | Timesteps between visualization frames |
| `--num_generations` | 5 | Number of images to generate |
| `--num_keep_checkpoints` | 1 | Number of recent checkpoints to keep |
| `--use_wandb` | False | Enable Weights & Biases logging |

## Training Process

The training loop implements the DDPM objective:

1. Sample random timesteps `t` for each image in the batch
2. Add noise to images according to the forward diffusion process
3. Predict the noise using the UNet model conditioned on timestep `t`
4. Compute loss between predicted and actual noise
5. Update model parameters using AdamW optimizer with cosine learning rate schedule

### Gradient Accumulation

The implementation supports gradient accumulation to simulate larger batch sizes:

```bash
# Effective batch size = batch_size * gradient_accumulation_steps
--batch_size 16 --gradient_accumulation_steps 4  # Effective: 64
```

### Checkpoint Management

- Checkpoints are saved every `evaluation_interval` steps
- Only the most recent `num_keep_checkpoints` are retained
- Each checkpoint contains:
  - Model state
  - Optimizer state
  - Scheduler state
  - Current training step

## Sampling & Generation

During evaluation, the model generates images by:

1. Starting with random Gaussian noise
2. Iteratively denoising for `num_diffusion_timesteps` steps
3. Visualizing intermediate denoising steps at specified intervals

Generated samples are saved as grids showing the progression from noise to final image.

## Model Architecture Details

### UNet Configuration

```
Encoder:
  - Initial Conv2d projection (3 → start_dim)
  - Residual blocks with channel progression (1x, 2x, 3x, 4x)
  - Downsampling by factor of 2 between stages
  - Self-attention after each downsampling

Bottleneck:
  - Multiple residual blocks at maximum dimension

Decoder:
  - Mirror of encoder with skip connections
  - Upsampling by factor of 2 between stages
  - Self-attention after each upsampling
  - Final Conv2d projection (start_dim → 3)
```

### Time Embedding

- Sinusoidal positional embeddings (similar to Transformers)
- MLP projection to match model dimensions
- Injected into each residual block

## Performance Optimization

### Apple Silicon (MPS)
- Automatic detection and usage of Metal Performance Shaders
- Manual attention implementation for compatibility
- Optimized memory usage with reduced default parameters

### Memory Management for Mac M3 Ultra
The implementation has been optimized for Mac M3 Ultra training:

**Memory-Efficient Settings:**
- **Starting Channels**: 96 (reduced from 128)
  - Prevents MPS memory overflow
  - Still maintains good model capacity
- **Batch Size**: 16
  - Optimal for 128x128 images on M3 Ultra
  - Can be increased for smaller images or reduced for larger
- **Gradient Accumulation**: Available to simulate larger batches
  - Example: `--batch_size 8 --gradient_accumulation_steps 2` = effective batch size 16

**Scaling Guidelines:**
| Image Size | Recommended Batch Size | Starting Channels |
|------------|------------------------|-------------------|
| 64x64      | 32-64                  | 128               |
| 128x128    | 16-32                  | 96-128            |
| 256x256    | 4-8                    | 64-96             |

### DataLoader Optimization
- **Persistent Workers**: Enabled for faster data loading
- **Pin Memory**: Disabled for MPS (set to False)
- **Num Workers**: 4 recommended for Mac
- **Prefetch Factor**: Automatic (handled by PyTorch)

### Training Speed Tips
1. Use SSD for dataset storage
2. Pre-resize images to target resolution
3. Enable `persistent_workers` in DataLoader
4. Monitor memory usage with Activity Monitor
5. Close unnecessary applications during training

## Results

Generated samples will be saved in the `generated_directory` at each evaluation interval. Each visualization shows:
- Multiple image generations (rows)
- Denoising progression from noise to final image (columns)
- Timestep intervals controlled by `plot_freq_interval`

## Citation

If you use this implementation in your research, please cite the original DDPM paper:

```bibtex
@inproceedings{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original DDPM paper by Ho et al.
- PyTorch team for the deep learning framework
- Hugging Face for the transformers library

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Issues

If you encounter any problems or have questions, please open an issue on GitHub.

### Common Issues & Solutions

#### MPS Memory Errors
```
RuntimeError: MPS backend out of memory
```
**Solutions:**
- Reduce `--batch_size` (try 8 or 12)
- Reduce `--starting_channels` (try 64 or 80)
- Reduce `--img_size` (try 64 or 96)
- Enable gradient accumulation: `--gradient_accumulation_steps 2`

#### Slow Data Loading
```
DataLoader warnings or slow iteration
```
**Solutions:**
- Reduce `--num_workers` (try 2)
- Ensure dataset is on SSD
- Pre-process images to target size
- Use `persistent_workers=True` (already enabled)



