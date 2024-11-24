#!/bin/bash

# Check if Python 3.10 is installed
if ! python3.10 --version &>/dev/null; then
    echo "Python 3.10 is required but not installed. Please install Python 3.10."
    exit 1
fi

# Install python3.10-venv if not installed
if ! dpkg -l | grep python3.10-venv &>/dev/null; then
    echo "Installing python3.10-venv..."
    sudo apt update
    sudo apt install -y python3.10-venv
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3.10 -m venv venv
fi

source venv/bin/activate

# Install cmake if not installed
if ! command -v cmake &>/dev/null; then
    echo "Installing cmake..."
    sudo apt install -y cmake
fi

# Install unrar if not installed
if ! command -v unrar &>/dev/null; then
    echo "Installing unrar..."
    sudo apt update
    sudo apt install -y unrar
fi

# Install PyTorch with CUDA 11.8 support
pip install --no-cache-dir torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install --no-cache-dir -r requirements.txt

# Add NVIDIA package repositories
sudo apt-key del 7fa2af80 2>/dev/null
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA Toolkit 11.8 and cuDNN 8 libraries
sudo apt-get -y install cuda-toolkit-11-8
sudo apt-get install -y libcudnn8=8.7.0.*-1+cuda11.8 libcudnn8-dev=8.7.0.*-1+cuda11.8

# Update environment variables
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

echo "Environment setup complete."
exec bash --rcfile <(echo "source venv/bin/activate")
