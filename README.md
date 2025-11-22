# ACT Policy C++ Inference

A standalone C++ implementation for running ACT (Action Chunking with Transformers) policy inference using LibTorch and OpenCV. This package includes everything needed to compile and run ACT inference on WSL with X Server visualization.

## 📦 Contents

- `act_inference.cpp` - Main C++ inference code
- `setup_wsl.sh` - Automated WSL environment setup script
- `compile.sh` - Compilation script
- `run_inference.sh` - Runtime script with X Server configuration
- `export_to_torchscript.py` - Python script to export PyTorch model to TorchScript
- `checkpoint/` - **Pre-trained ACT model checkpoint** (ready to use!)
- `README.md` - This file

## 🚀 Quick Start

### Prerequisites

1. **WSL2** installed on Windows
2. **VcXsrv** X Server for Windows (download from [SourceForge](https://sourceforge.net/projects/vcxsrv/))
3. **Trained ACT model** checkpoint directory
4. **Dataset** with videos and stats.json

### Step 1: Setup VcXsrv (Windows)

1. Download and install VcXsrv
2. Launch with these settings:
   - **Multiple windows**
   - **Display number: 0**
   - **Start no client**
   - ✅ **Check "Disable access control"** (important!)

### Step 2: Export Model to TorchScript (Optional)

**Note:** This package includes a pre-trained checkpoint with `model_torchscript.pt` already exported! You can skip this step if using the included checkpoint.

If you want to use your own checkpoint, export it to TorchScript:

```bash
# From your Python environment with lerobot installed
python export_to_torchscript.py <checkpoint_dir>

# Example:
python export_to_torchscript.py /mnt/d/lerobot/outputs/train/.../checkpoints/300000
```

This creates `model_torchscript.pt` in `checkpoint_dir/pretrained_model/`

### Step 3: Setup WSL Environment

```bash
cd act_inference_cpp_standalone
chmod +x *.sh
bash setup_wsl.sh
```

This will:
- Configure X Server display
- Install OpenCV, LibTorch, and dependencies
- Install nlohmann-json
- Test X Server connection

### Step 4: Compile

```bash
bash compile.sh
```

### Step 5: Run Inference

**Using the included checkpoint:**

```bash
# If running from the package directory in WSL:
bash run_inference.sh ./checkpoint <dataset_dir>

# Example with full paths:
bash run_inference.sh \
  /mnt/d/path/to/act_inference_cpp_standalone/checkpoint \
  /mnt/d/lerobot/dataset/Marker_pickup_piper
```

**Using your own checkpoint:**

```bash
bash run_inference.sh <checkpoint_dir> <dataset_dir>
```

## 🎮 Controls

- **`q`** or **ESC** - Quit
- **`p`** - Pause/Resume

## 📁 Required Directory Structure

Your checkpoint and dataset directories should have this structure:

```
checkpoint_dir/
└── pretrained_model/
    ├── config.json
    ├── model.safetensors
    └── model_torchscript.pt  (created by export script)

dataset_dir/
├── meta/
│   └── stats.json
└── videos/
    ├── observation.images.main/
    │   └── chunk-000/
    │       └── file-000.mp4
    └── observation.images.secondary_0/
        └── chunk-000/
            └── file-000.mp4
```

## 🔧 Manual Compilation

If the scripts don't work, compile manually:

```bash
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0
export LD_LIBRARY_PATH=/usr/local/libtorch/lib:$LD_LIBRARY_PATH

g++ -std=c++17 act_inference.cpp \
    -I/usr/local/libtorch/include \
    -I/usr/local/libtorch/include/torch/csrc/api/include \
    -L/usr/local/libtorch/lib \
    $(pkg-config --cflags --libs opencv4) \
    -ltorch -ltorch_cpu -lc10 \
    -Wl,-rpath,/usr/local/libtorch/lib \
    -O3 -o act_inference
```

## 🐛 Troubleshooting

### "Cannot open display"
- Make sure VcXsrv is running
- Check firewall isn't blocking VcXsrv
- Verify DISPLAY is set: `echo $DISPLAY`
- Try: `export DISPLAY=:0` or `export DISPLAY=localhost:0`

### "LibTorch not found"
- Run `setup_wsl.sh` first
- Or manually install LibTorch to `/usr/local/libtorch`

### "Model file not found"
- Make sure you ran `export_to_torchscript.py` first
- Check that `model_torchscript.pt` exists in `checkpoint_dir/pretrained_model/`

### "Stats file not found"
- Ensure `dataset_dir/meta/stats.json` exists
- This file is created during dataset preparation

### Test X Server Connection

```bash
xclock
```

If you see a clock window on Windows, X Server is working! 🎉

## 📝 Notes

- **Included checkpoint:** This package includes a pre-trained ACT checkpoint in the `checkpoint/` directory. The TorchScript model (`model_torchscript.pt`) is already exported and ready to use.
- The code uses **WSL path format** (`/mnt/d/...`) by default
- For **CUDA support**, modify `setup_wsl.sh` to download CUDA version of LibTorch
- The visualization shows:
  - Main and secondary camera feeds
  - Predicted action trajectories for 7 motors
  - Current state and ground truth (if available)
  - Real-time FPS

## 📄 License

Same as the parent lerobot project.

## 🤝 Contributing

This is a standalone package. For issues or improvements, please refer to the main lerobot repository.

