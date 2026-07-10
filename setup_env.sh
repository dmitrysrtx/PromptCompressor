#!/bin/bash

echo "🚀 Starting environment setup..."

# 1. Hardcode Git settings (to avoid being prompted each time)
git config --global user.email "<your@email>"
git config --global user.name "<yourusername>"
echo "✅ Git configured"


# 2. RL environment (Python 3.9)
echo "==============================================="
echo "🚀 Starting RL environment setup (Python 3.9)..."
echo "==============================================="

echo "📦 Step 1: Downloading Python 3.9..."
apt-get update -qq
apt-get install python3.9 python3.9-venv python3.9-dev -y > /dev/null 2>&1

echo "🛠️ Step 2: Creating virtual environment (/content/env39)..."
python3.9 -m venv /content/env39

echo "🔄 Step 3: Activating environment..."
# In bash script need to use full source command
source /content/env39/bin/activate

echo "⚙️ Step 4: Downgrading pip and installing basic build tools..."
pip install pip==23.0.1
pip install setuptools==65.5.0 wheel==0.38.4

echo "🏋️ Step 5: Installing finicky gym and stable-baselines3..."
pip install gym==0.21.0 --no-use-pep517
pip install stable-baselines3==1.8.0 --no-use-pep517

echo "📚 Step 6: Installing requirements.txt (this will take a couple of minutes)..."
pip install -r requirements.txt

# 3. Load secret API key from .env file (if it exists)
if [ -f ".env" ]; then
    export $(cat .env | xargs)
    echo "✅ Secrets (W&B API Key) loaded successfully"
else
    echo "⚠️ .env file not found! W&B authentication might not work."
fi

echo "==============================================="
echo "✅ DONE! Environment successfully set up."
echo "==============================================="
