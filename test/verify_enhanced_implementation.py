import torch
import sys
from configs.gazegaussian_options import BaseOptions

print("=" * 80)
print("ENHANCED GAZEGAUSSIAN IMPLEMENTATION VERIFICATION")
print("=" * 80)

opts = BaseOptions()

enhancements = {
    "AdaLN-Zero in DiT Blocks": False,
    "VAE Integration": False,
    "Orthogonality Loss": False,
    "Intermediate Gaze Training": False,
}

print("\n[1/4] Verifying AdaLN-Zero Enhancement...")
print("-" * 80)

try:
    from models.dit_block import DiTBlock, AdaptiveLayerNorm
    
    test_block = DiTBlock(dim=512, num_heads=8, cond_dim=64)
    
    if hasattr(test_block, 'adaLN'):
        adaln = test_block.adaLN
        if hasattr(adaln, 'adaLN_modulation'):
            print("✓ AdaLN-Zero modulation found")
            
            x = torch.randn(2, 256, 512)
            cond = torch.randn(2, 64)
            
            with torch.no_grad():
                output = adaln(x, cond)
                
            if len(output) == 6:
                print("✓ Generates 6 parameters (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)")
                enhancements["AdaLN-Zero in DiT Blocks"] = True
            else:
                print(f"✗ Expected 6 outputs, got {len(output)}")
        else:
            print("✗ adaLN_modulation not found")
    else:
        print("✗ adaLN attribute not found in DiTBlock")
        
except Exception as e:
    print(f"✗ Error testing AdaLN-Zero: {e}")

print("\n[2/4] Verifying VAE Integration...")
print("-" * 80)

if hasattr(opts, 'use_vae'):
    print(f"✓ use_vae config found: {opts.use_vae}")
    print(f"✓ vae_z_channels: {opts.vae_z_channels}")
    print(f"✓ freeze_vae: {opts.freeze_vae}")
    
    try:
        from models.vae_module import VAE, VAEEncoder, VAEDecoder
        print("✓ VAE module components imported")
        
        vae = VAE(in_channels=256, out_channels=256, z_channels=4)
        x = torch.randn(2, 256, 128, 128)
        
        with torch.no_grad():
            mean, logvar = vae.encode(x)
            z = mean
            output = vae.decode(z)
            
        print(f"✓ VAE encode: {x.shape} → {mean.shape}")
        print(f"✓ VAE decode: {z.shape} → {output.shape}")
        
        if output.shape == x.shape:
            print("✓ VAE reconstruction shape matches")
            enhancements["VAE Integration"] = True
        else:
            print(f"✗ Shape mismatch: {output.shape} vs {x.shape}")
            
    except Exception as e:
        print(f"✗ Error testing VAE: {e}")
else:
    print("✗ use_vae config not found")

print("\n[3/4] Verifying Orthogonality Loss...")
print("-" * 80)

if hasattr(opts, 'use_orthogonality_loss') and opts.use_orthogonality_loss:
    print(f"✓ Orthogonality Loss enabled: importance={opts.orthogonality_loss_importance}")
    
    try:
        from losses.gazenerf_loss import GazeNeRFLoss
        
        loss_func = GazeNeRFLoss(
            gaze_loss_importance=0.1,
            eye_loss_importance=1.0,
            face_loss_importance=1.0,
            vgg_importance=0.1,
            device='cpu'
        )
        
        if hasattr(loss_func, 'calc_orthogonality_loss'):
            print("✓ calc_orthogonality_loss method found")
            
            gaze_feat = torch.randn(2, 2)
            head_feat = torch.randn(2, 2)
            expr_feat = torch.randn(2, 2)
            
            orth_loss = loss_func.calc_orthogonality_loss(gaze_feat, head_feat, expr_feat)
            print(f"✓ Orthogonality loss computed: {orth_loss.item():.6f}")
            enhancements["Orthogonality Loss"] = True
        else:
            print("✗ calc_orthogonality_loss not found")
            
    except Exception as e:
        print(f"✗ Error: {e}")
else:
    print("✗ Orthogonality Loss not enabled")

print("\n[4/4] Verifying Intermediate Gaze Training...")
print("-" * 80)

if hasattr(opts, 'use_intermediate_gaze_training') and opts.use_intermediate_gaze_training:
    print(f"✓ Intermediate Gaze Training enabled")
    print(f"  - Num intermediate gazes: {opts.num_intermediate_gazes}")
    print(f"  - Training frequency: every {opts.intermediate_gaze_freq} batches")
    
    try:
        from trainer.gazegaussian_trainer import GazeGaussianTrainer
        
        if hasattr(GazeGaussianTrainer, 'generate_intermediate_gaze_angles'):
            print("✓ generate_intermediate_gaze_angles method found")
            enhancements["Intermediate Gaze Training"] = True
        else:
            print("✗ generate_intermediate_gaze_angles not found")
            
    except Exception as e:
        print(f"✗ Error: {e}")
else:
    print("✗ Intermediate Gaze Training not enabled")

print("\n" + "=" * 80)
print("ENHANCEMENT SUMMARY")
print("=" * 80)

all_enhanced = all(enhancements.values())

for enhancement, status in enhancements.items():
    symbol = "✓" if status else "✗"
    status_text = "ACTIVE" if status else "FAILED"
    print(f"{symbol} {enhancement}: {status_text}")

print("\n" + "=" * 80)

if all_enhanced:
    print("✅ ALL ENHANCEMENTS SUCCESSFULLY INTEGRATED!")
    print("\nYour implementation now includes:")
    print("  1. AdaLN-Zero - Enhanced adaptive conditioning with gate parameters")
    print("  2. VAE Integration - Latent space compression and reconstruction")
    print("  3. Orthogonality Loss - Disentangled feature control")
    print("  4. Intermediate Gaze Training - Improved gaze consistency")
    print("\nConfiguration:")
    print(f"  • DiT Neural Renderer: {opts.neural_renderer_type}")
    print(f"  • Use VAE: {opts.use_vae}")
    print(f"  • DiT Depth: {opts.dit_depth}")
    print(f"  • Orthogonality Loss: {opts.use_orthogonality_loss}")
    print(f"  • Intermediate Gaze: {opts.use_intermediate_gaze_training}")
    print("\n🚀 Ready for high-fidelity gaze redirection training!")
else:
    failed = [k for k, v in enhancements.items() if not v]
    print(f"⚠ {len(failed)} enhancement(s) not fully operational:")
    for enh in failed:
        print(f"  - {enh}")
    print("\nCheck error messages above for details.")

print("=" * 80 + "\n")

sys.exit(0 if all_enhanced else 1)
