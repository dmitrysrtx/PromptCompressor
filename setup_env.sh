#!/bin/bash

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

echo "==============================================="
echo "✅ ГОТОВО! Окружение успешно собрано."
echo "==============================================="
