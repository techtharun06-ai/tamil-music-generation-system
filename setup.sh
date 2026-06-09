#!/bin/bash

echo "🎵 Setting up Tamil Music Generation System (Model 4)..."
echo "========================================================="

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}[1/6]${NC} Creating virtual environment..."
python -m venv venv
source venv/bin/activate

echo -e "${BLUE}[2/6]${NC} Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo -e "${BLUE}[3/6]${NC} Installing PyTorch and core dependencies..."
pip install torch==2.0.0 torchaudio==2.0.0 torchvision==0.15.0

echo -e "${BLUE}[4/6]${NC} Installing all dependencies..."
pip install -r requirements.txt --no-cache-dir

echo -e "${BLUE}[5/6]${NC} Creating project directories..."
mkdir -p data/raw/audio data/raw/lyrics
mkdir -p data/processed/{stems,features,metadata}
mkdir -p outputs/{checkpoints/{stage_a,stage_b},generated_audio,logs}
mkdir -p config notebooks scripts src/{data,models,training,inference,evaluation,utils}

echo -e "${BLUE}[6/6]${NC} Downloading Demucs model..."
python -c "import demucs; print('✓ Demucs ready')"

echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Download dataset:  python scripts/download_dataset.py"
echo "2. Preprocess data:   python scripts/preprocess_data.py"
echo "3. Train Stage A:     python scripts/train_stage_a.py"
echo "4. Train Stage B:     python scripts/train_stage_b.py"
echo ""
