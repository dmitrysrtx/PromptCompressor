#!/bin/bash

echo "🚀 Начинаем настройку окружения..."

# 1. Жестко прописываем настройки Git (чтобы не просил каждый раз)
git config --global user.email "dima.strizhak@gmail.com"
git config --global user.name "dmitrysrtx"
echo "✅ Git настроен"


# 2. RL-окружение (Python 3.9)
echo "==============================================="
echo "🚀 Начинаем сборку RL-окружения (Python 3.9)..."
echo "==============================================="

echo "📦 Шаг 1: Скачиваем Python 3.9..."
apt-get update -qq
apt-get install python3.9 python3.9-venv python3.9-dev -y > /dev/null 2>&1

echo "🛠️ Шаг 2: Создаем виртуальную среду (/content/env39)..."
python3.9 -m venv /content/env39

echo "🔄 Шаг 3: Активируем среду..."
# В bash-скрипте нужно использовать полную команду source
source /content/env39/bin/activate

echo "⚙️ Шаг 4: Откатываем pip и ставим базовые сборщики..."
pip install pip==23.0.1
pip install setuptools==65.5.0 wheel==0.38.4

echo "🏋️ Шаг 5: Устанавливаем капризные gym и stable-baselines3..."
pip install gym==0.21.0 --no-use-pep517
pip install stable-baselines3==1.8.0 --no-use-pep517

echo "📚 Шаг 6: Устанавливаем requirements.txt (это займет пару минут)..."
pip install -r requirements.txt
pip install --upgrade wandb

# 3. Подгружаем секретный API-ключ из файла .env (если он существует)
if [ -f ".env" ]; then
    export $(cat .env | xargs)
    echo "✅ Секреты (W&B API Key) успешно загружены"
else
    echo "⚠️ Файл .env не найден! Авторизация W&B может не сработать."
fi

echo "==============================================="
echo "✅ ГОТОВО! Окружение успешно собрано."
echo "==============================================="
