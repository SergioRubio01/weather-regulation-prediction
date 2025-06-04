#!/bin/bash
# Script to install CUDA 12.5 in WSL2 for TensorFlow GPU support

echo "=== Installing CUDA 12.5 for WSL2 ==="
echo ""

# Step 1: Remove old CUDA packages if any
echo "1. Removing old CUDA packages..."
sudo apt-get --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" 2>/dev/null

# Step 2: Add NVIDIA package repository
echo "2. Adding NVIDIA package repository..."
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Step 3: Install CUDA toolkit
echo "3. Installing CUDA 12.5..."
sudo apt-get -y install cuda-toolkit-12-5

# Step 4: Set up environment variables
echo "4. Setting up environment variables..."
echo "" >> ~/.bashrc
echo "# CUDA configuration" >> ~/.bashrc
echo "export CUDA_HOME=/usr/local/cuda-12.5" >> ~/.bashrc
echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc

echo ""
echo "=== CUDA 12.5 Installation Complete ==="
echo ""
echo "Next steps:"
echo "1. Download cuDNN 9.5 from: https://developer.nvidia.com/cudnn (requires NVIDIA account)"
echo "2. Extract and copy cuDNN files to /usr/local/cuda-12.5/"
echo "3. Restart your terminal or run: source ~/.bashrc"
echo "4. Test with: python -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'"
