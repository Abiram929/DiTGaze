import sys
from configs.gazegaussian_options import BaseOptions

print("=" * 80)
print("DiT Neural Renderer Integration Verification")
print("=" * 80)

opts = BaseOptions()

print(f"\n✓ Config loaded successfully")
print(f"  Neural renderer type: {opts.neural_renderer_type}")
print(f"  DiT depth: {opts.dit_depth}")
print(f"  DiT num heads: {opts.dit_num_heads}")
print(f"  DiT patch size: {opts.dit_patch_size}")

try:
    from models.dit_block import DiTBlock, PatchEmbed, Unpatchify
    print(f"\n✓ DiT block module imported successfully")
except Exception as e:
    print(f"\n✗ Failed to import DiT block: {e}")
    sys.exit(1)

try:
    from models.dit_neural_renderer import DiTNeuralRenderer
    print(f"✓ DiT neural renderer module imported successfully")
except Exception as e:
    print(f"✗ Failed to import DiT neural renderer: {e}")
    print(f"  Note: kornia library required (pip install kornia)")
    sys.exit(1)

try:
    from models.gaze_gaussian import GazeGaussianNet
    print(f"✓ GazeGaussianNet imported successfully")
except Exception as e:
    print(f"✗ Failed to import GazeGaussianNet: {e}")
    sys.exit(1)

print(f"\n{'='*80}")
print("Integration Summary")
print(f"{'='*80}")

print(f"\nFiles created:")
print(f"  1. models/dit_block.py - DiT transformer components")
print(f"  2. models/dit_neural_renderer.py - DiT-based neural renderer")

print(f"\nFiles modified:")
print(f"  1. configs/gazegaussian_options.py - Added DiT configuration")
print(f"  2. models/gaze_gaussian.py - Added conditional renderer selection")

print(f"\nConfiguration options added:")
print(f"  - neural_renderer_type: 'dit' or 'progressive'")
print(f"  - dit_depth: {opts.dit_depth} (number of transformer blocks)")
print(f"  - dit_num_heads: {opts.dit_num_heads} (attention heads)")
print(f"  - dit_patch_size: {opts.dit_patch_size} (patch size for tokenization)")
print(f"  - dit_mlp_ratio: {opts.dit_mlp_ratio} (MLP expansion ratio)")

print(f"\nHow to use:")
print(f"  1. Set neural_renderer_type='dit' in config (already set)")
print(f"  2. Train normally - DiT will be used automatically")
print(f"  3. To use original renderer, set neural_renderer_type='progressive'")

print(f"\n✓ All integration checks passed!")
print(f"{'='*80}\n")
