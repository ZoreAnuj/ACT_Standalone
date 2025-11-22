#!/bin/bash
# ============================================
# WSL Setup for ACT Inference with X Server
# ============================================

echo "Setting up WSL for ACT Inference with X Server visualization..."
echo ""

# Step 1: Set DISPLAY variable
echo "Step 1: Configuring DISPLAY variable..."
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0
echo "export DISPLAY=\$(cat /etc/resolv.conf | grep nameserver | awk '{print \$2}'):0.0" >> ~/.bashrc
echo "DISPLAY set to: $DISPLAY"
echo ""

# Step 2: Install build dependencies
echo "Step 2: Installing build dependencies..."
sudo apt update
sudo apt install -y build-essential cmake git wget unzip pkg-config
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y libx11-dev libxext-dev libxrender-dev libxinerama-dev libxi-dev libxrandr-dev libxcursor-dev libxtst-dev libgtk2.0-dev
echo ""

# Step 3: Install LibTorch
echo "Step 3: Installing LibTorch..."
if [ ! -d "/usr/local/libtorch" ]; then
    cd /tmp
    echo "Downloading LibTorch (CPU version)..."
    wget -q --show-progress https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip -O libtorch.zip
    
    echo "Extracting..."
    unzip -q libtorch.zip
    sudo mv libtorch /usr/local/
    rm libtorch.zip
    echo "LibTorch installed to /usr/local/libtorch"
else
    echo "LibTorch already installed"
fi
echo ""

# Step 4: Install nlohmann-json
echo "Step 4: Installing nlohmann-json..."
sudo apt install -y nlohmann-json3-dev
echo ""

# Step 5: Test X Server connection
echo "Step 5: Testing X Server connection..."
sudo apt install -y x11-apps
echo "Launching test window (xclock)..."
xclock &
XCLOCK_PID=$!
sleep 2
kill $XCLOCK_PID 2>/dev/null
echo ""

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Make sure VcXsrv is running on Windows"
echo "2. Compile: bash compile.sh"
echo "3. Run: bash run_inference.sh"
echo ""

