#!/bin/bash
# ============================================
# Compile ACT Inference C++ Code
# ============================================

echo "Compiling ACT Inference..."
echo ""

# Set LibTorch path
LIBTORCH_PATH="/usr/local/libtorch"

# Check if LibTorch exists
if [ ! -d "$LIBTORCH_PATH" ]; then
    echo "Error: LibTorch not found at $LIBTORCH_PATH"
    echo "Please run setup_wsl.sh first"
    exit 1
fi

# Compile with g++
g++ -std=c++17 act_inference.cpp \
    -I${LIBTORCH_PATH}/include \
    -I${LIBTORCH_PATH}/include/torch/csrc/api/include \
    -L${LIBTORCH_PATH}/lib \
    $(pkg-config --cflags --libs opencv4) \
    -ltorch -ltorch_cpu -lc10 \
    -Wl,-rpath,${LIBTORCH_PATH}/lib \
    -O3 -o act_inference

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Compilation successful!"
    echo "=========================================="
    echo ""
    echo "Run with: bash run_inference.sh"
    echo "Or with custom paths:"
    echo "  ./act_inference <checkpoint_dir> <dataset_dir>"
    echo ""
else
    echo ""
    echo "Compilation failed!"
    exit 1
fi

