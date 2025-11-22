"""
Export ACT Policy to TorchScript for C++ Inference
This script exports the trained PyTorch model to TorchScript format.

Usage:
    python export_to_torchscript.py <checkpoint_dir>
    
Example:
    python export_to_torchscript.py /path/to/checkpoint/300000
"""

import json
import torch
from pathlib import Path
import sys

# Try to import lerobot - user needs to have it installed or in path
try:
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.configs.types import PolicyFeature, FeatureType
    from safetensors.torch import load_file
except ImportError as e:
    print("Error: Could not import lerobot modules.")
    print("Please ensure lerobot is installed or add it to your Python path.")
    print(f"Import error: {e}")
    sys.exit(1)


class ACTPolicyWrapper(torch.nn.Module):
    """Wrapper to make ACTPolicy compatible with TorchScript"""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        
    def forward(self, batch_dict):
        """
        Forward pass that accepts a dictionary
        batch_dict should contain:
            - 'observation.state': (B, state_dim)
            - 'observation.images.main': (B, C, H, W)
            - 'observation.images.secondary_0': (B, C, H, W)
        """
        # Run inference
        with torch.no_grad():
            predicted_actions = self.policy.predict_action_chunk(batch_dict)
        return predicted_actions


def export_model_to_torchscript(checkpoint_dir: str, output_path: str = None):
    """
    Export ACT model to TorchScript format
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        output_path: Output path for TorchScript model (optional)
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if output_path is None:
        output_path = checkpoint_dir / 'pretrained_model' / 'model_torchscript.pt'
    
    print(f"Loading model from: {checkpoint_dir}")
    
    # Load configuration
    config_path = checkpoint_dir / 'pretrained_model' / 'config.json'
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert feature dicts to PolicyFeature objects
    if 'input_features' in config_dict:
        input_features = {}
        for key, feat_dict in config_dict['input_features'].items():
            input_features[key] = PolicyFeature(
                type=FeatureType[feat_dict['type']],
                shape=tuple(feat_dict['shape'])
            )
        config_dict['input_features'] = input_features
    
    if 'output_features' in config_dict:
        output_features = {}
        for key, feat_dict in config_dict['output_features'].items():
            output_features[key] = PolicyFeature(
                type=FeatureType[feat_dict['type']],
                shape=tuple(feat_dict['shape'])
            )
        config_dict['output_features'] = output_features
    
    # Remove keys that aren't ACTConfig parameters
    keys_to_remove = ['type', 'device', 'use_amp', 'push_to_hub', 'repo_id', 
                     'private', 'tags', 'license', 'pretrained_path']
    for key in keys_to_remove:
        config_dict.pop(key, None)
    
    # Create ACTConfig and policy
    config = ACTConfig(**config_dict)
    policy = ACTPolicy(config)
    policy.eval()
    
    # Load weights
    print("Loading weights...")
    weights_path = checkpoint_dir / 'pretrained_model' / 'model.safetensors'
    if not weights_path.exists():
        print(f"Error: Weights file not found at {weights_path}")
        return False
    
    weights = load_file(weights_path)
    policy.load_state_dict(weights, strict=False)
    policy.to('cpu')  # Export on CPU for compatibility
    
    print("Model loaded successfully!")
    
    # Create wrapper
    wrapped_model = ACTPolicyWrapper(policy)
    wrapped_model.eval()
    
    # Create example inputs
    print("Creating example inputs...")
    batch_size = 1
    state_dim = config_dict['input_features']['observation.state']['shape'][0]
    
    example_batch = {
        'observation.state': torch.randn(batch_size, state_dim),
        'observation.images.main': torch.randn(batch_size, 3, 240, 320),
        'observation.images.secondary_0': torch.randn(batch_size, 3, 240, 320),
    }
    
    print("Tracing model with TorchScript...")
    try:
        # Use torch.jit.trace
        traced_model = torch.jit.trace(wrapped_model, (example_batch,))
        
        # Save the model
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving TorchScript model to: {output_path}")
        traced_model.save(str(output_path))
        
        print("✓ Model successfully exported to TorchScript!")
        print(f"  Output: {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Verify the model can be loaded
        print("\nVerifying exported model...")
        loaded_model = torch.jit.load(str(output_path))
        test_output = loaded_model(example_batch)
        print(f"✓ Model verification successful! Output shape: {test_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during export: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: TorchScript tracing may fail if the model contains dynamic control flow.")
        print("In that case, you may need to modify the model or use torch.jit.script instead.")
        return False


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python export_to_torchscript.py <checkpoint_dir>")
        print("\nExample:")
        print("  python export_to_torchscript.py /path/to/checkpoint/300000")
        sys.exit(1)
    
    checkpoint_dir = sys.argv[1]
    
    print("=" * 60)
    print("ACT Model Export to TorchScript")
    print("=" * 60)
    print()
    
    success = export_model_to_torchscript(checkpoint_dir)
    
    if success:
        print("\n" + "=" * 60)
        print("Export complete! You can now use the C++ inference code.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Export failed. Please check the errors above.")
        print("=" * 60)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

