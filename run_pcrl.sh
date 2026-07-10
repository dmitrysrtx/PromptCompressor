#!/bin/bash

# Navigate to the project working directory
cd /content/drive/MyDrive/StudiesAI/RL/PromptCompressor || { echo "❌ Error: Failed to navigate to project directory"; exit 1; }

# 1. System output flags
export PYTHONUNBUFFERED=1
export MPLBACKEND="Agg"

# 2. Set Hugging Face to ONLINE mode (remove or set to 0)
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# 3. Keep WandB flag ONLINE!
# This is what protects against hangs during authorization
export WANDB_MODE="online"

echo "🚀 Launching PCRL in fully isolated offline mode from terminal..."

# 4. Activate virtual environment and run Python directly
source /content/env39/bin/activate

python train_pcrl.py \
    --config_path configs/gpt2-xl-code.yml \
    --project_name PromptCompressor_code_RL \
    --experiment_name gpt2_xl_code_v1_bleu_512 \
    --log_to_wandb