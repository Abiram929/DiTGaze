import sys
import torch
from configs.gazegaussian_options import BaseOptions

print("=" * 80)
print("VERIFICATION: 3 Contributions Integration")
print("=" * 80)

opts = BaseOptions()

contribution_status = {
    "DiT Neural Renderer": False,
    "Orthogonality Loss": False,
    "Intermediate Gaze Angles Training": False
}

print("\n[1/3] Checking DiT Neural Renderer...")
print("-" * 80)

if hasattr(opts, 'neural_renderer_type'):
    print(f"✓ neural_renderer_type found: {opts.neural_renderer_type}")
    if opts.neural_renderer_type == "dit":
        print(f"✓ DiT is ACTIVE")
        print(f"  - DiT depth: {opts.dit_depth}")
        print(f"  - DiT num heads: {opts.dit_num_heads}")
        print(f"  - DiT patch size: {opts.dit_patch_size}")
        contribution_status["DiT Neural Renderer"] = True
    else:
        print(f"⚠ DiT is configured but set to: {opts.neural_renderer_type}")
        print(f"  Change to 'dit' in config to activate")
else:
    print("✗ neural_renderer_type not found in config")

try:
    from models.dit_neural_renderer import DiTNeuralRenderer
    print("✓ DiTNeuralRenderer module imported successfully")
except Exception as e:
    print(f"✗ Failed to import DiTNeuralRenderer: {e}")

print("\n[2/3] Checking Orthogonality Loss...")
print("-" * 80)

if hasattr(opts, 'use_orthogonality_loss'):
    print(f"✓ use_orthogonality_loss found: {opts.use_orthogonality_loss}")
    if opts.use_orthogonality_loss:
        print(f"✓ Orthogonality Loss is ACTIVE")
        print(f"  - Loss importance: {opts.orthogonality_loss_importance}")
        contribution_status["Orthogonality Loss"] = True
    else:
        print(f"⚠ Orthogonality Loss is configured but disabled")
        print(f"  Set use_orthogonality_loss=True in config to activate")
else:
    print("✗ use_orthogonality_loss not found in config")

try:
    from losses.gazenerf_loss import GazeNeRFLoss
    loss = GazeNeRFLoss(
        gaze_loss_importance=0.1,
        eye_loss_importance=1.0,
        face_loss_importance=1.0,
        vgg_importance=0.1,
        device='cpu'
    )
    
    if hasattr(loss, 'calc_orthogonality_loss'):
        print("✓ calc_orthogonality_loss method found in GazeNeRFLoss")
        
        gaze_feat = torch.randn(2, 2)
        head_feat = torch.randn(2, 3)
        expr_feat = torch.randn(2, 64)
        orth_loss = loss.calc_orthogonality_loss(gaze_feat, head_feat, expr_feat)
        print(f"✓ Orthogonality loss computation test passed: {orth_loss.item():.6f}")
    else:
        print("✗ calc_orthogonality_loss method not found")
except Exception as e:
    print(f"✗ Error testing orthogonality loss: {e}")

print("\n[3/3] Checking Intermediate Gaze Angles Training...")
print("-" * 80)

if hasattr(opts, 'use_intermediate_gaze_training'):
    print(f"✓ use_intermediate_gaze_training found: {opts.use_intermediate_gaze_training}")
    if opts.use_intermediate_gaze_training:
        print(f"✓ Intermediate Gaze Training is ACTIVE")
        print(f"  - Num intermediate gazes: {opts.num_intermediate_gazes}")
        print(f"  - Training frequency: every {opts.intermediate_gaze_freq} batches")
        contribution_status["Intermediate Gaze Angles Training"] = True
    else:
        print(f"⚠ Intermediate Gaze Training is configured but disabled")
        print(f"  Set use_intermediate_gaze_training=True in config to activate")
else:
    print("✗ use_intermediate_gaze_training not found in config")

try:
    from trainer.gazegaussian_trainer import GazeGaussianTrainer
    
    if hasattr(GazeGaussianTrainer, 'generate_intermediate_gaze_angles'):
        print("✓ generate_intermediate_gaze_angles method found in trainer")
    else:
        print("✗ generate_intermediate_gaze_angles method not found")
except Exception as e:
    print(f"✗ Error checking trainer: {e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

all_active = all(contribution_status.values())

for contrib, status in contribution_status.items():
    symbol = "✓" if status else "✗"
    status_text = "ACTIVE" if status else "INACTIVE"
    print(f"{symbol} {contrib}: {status_text}")

print("\n" + "=" * 80)

if all_active:
    print("✓ ALL 3 CONTRIBUTIONS ARE ACTIVE AND INTEGRATED!")
    print("\nYour model now uses:")
    print("  1. DiT Neural Renderer for feature refinement")
    print("  2. Orthogonality Loss for disentangled control")
    print("  3. Intermediate Gaze Angles for improved gaze consistency")
    print("\nReady for training!")
else:
    inactive = [k for k, v in contribution_status.items() if not v]
    print(f"⚠ {len(inactive)} contribution(s) not fully active:")
    for contrib in inactive:
        print(f"  - {contrib}")
    print("\nCheck config settings and imports above.")

print("=" * 80 + "\n")

sys.exit(0 if all_active else 1)
