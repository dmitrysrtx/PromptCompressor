#!/bin/bash

# Переходим в рабочую директорию проекта
cd /content/drive/MyDrive/StudiesAI/RL/PromptCompressor || { echo "❌ Ошибка: Не удалось перейти в директорию проекта"; exit 1; }

# 1. Системные флаги вывода
export PYTHONUNBUFFERED=1
export MPLBACKEND="Agg"

# 2. Переводим Hugging Face в ОНЛАЙН режим (удаляем или меняем на 0)
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# 3. Бронебойный флаг для WandB ОСТАВЛЯЕМ В ОФЛАЙНЕ!
# Именно он защищает от зависаний на авторизации
export WANDB_MODE="online"

echo "🚀 Запуск PCRL в полностью изолированном автономном режиме из терминала..."

# 4. Активируем виртуальную среду и запускаем Python напрямую
source /content/env39/bin/activate

python -u train_pcrl.py \
    --config_path configs/gpt2-xl-code.yml \
    --project_name PromptCompressor_code_RL \
    --experiment_name gpt2_xl_code_scst_isolated \
    --log_to_wandb