#!/bin/bash
# ============================================
# Run ACT Inference with X Server
# ============================================

# Ensure DISPLAY is set
if [ -z "$DISPLAY" ]; then
    export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0
    echo "Set DISPLAY to: $DISPLAY"
fi

# Check if VcXsrv is accessible
if ! xdpyinfo &>/dev/null; then
    echo "Warning: Cannot connect to X Server!"
    echo ""
    echo "Make sure VcXsrv is running on Windows with:"
    echo "  - Display number: 0"
    echo "  - 'Disable access control' checked"
    echo ""
    echo "Trying to continue anyway..."
fi

# Set LibTorch library path
export LD_LIBRARY_PATH=/usr/local/libtorch/lib:$LD_LIBRARY_PATH

# Run the inference
echo "Starting ACT Inference..."
echo "Press 'q' to quit, 'p' to pause"
echo ""

./act_inference "$@"

