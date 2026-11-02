import torch
import sys
from configs.gazegaussian_options import BaseOptions

def test_dit_renderer():
    print("=" * 80)
    print("Testing DiT Neural Renderer")
    print("=" * 80)
    
    opts = BaseOptions()
    
    print(f"\nRenderer type: {opts.neural_renderer_type}")
    print(f"DiT depth: {opts.dit_depth}")
    print(f"DiT num heads: {opts.dit_num_heads}")
    print(f"DiT patch size: {opts.dit_patch_size}")
    print(f"Feature map size: {opts.featmap_size}")
    print(f"Image size: {opts.pred_img_size}")
    
    batch_size = 2
    feat_nc = opts.featmap_nc
    featmap_size = opts.featmap_size
    
    print(f"\n{'='*80}")
    print("Testing DiT Renderer Standalone")
    print(f"{'='*80}")
    
    from models.dit_neural_renderer import DiTNeuralRenderer
    
    dit_renderer = DiTNeuralRenderer(
        bg_type=opts.bg_type,
        feat_nc=feat_nc,
        out_dim=3,
        final_actvn=True,
        min_feat=32,
        featmap_size=featmap_size,
        img_size=opts.pred_img_size,
        patch_size=opts.dit_patch_size,
        depth=opts.dit_depth,
        num_heads=opts.dit_num_heads,
        mlp_ratio=opts.dit_mlp_ratio,
        cond_dim=opts.shape_coeffs_dim if opts.unet_atten else None,
    )
    
    x = torch.randn(batch_size, feat_nc, featmap_size, featmap_size)
    shape_code = torch.randn(batch_size, opts.shape_coeffs_dim) if opts.unet_atten else None
    
    print(f"\nInput shape: {x.shape}")
    if shape_code is not None:
        print(f"Condition shape: {shape_code.shape}")
    
    with torch.no_grad():
        output = dit_renderer(x, shape_code)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    total_params = sum(p.numel() for p in dit_renderer.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print(f"\n{'='*80}")
    print("Testing Progressive Renderer for Comparison")
    print(f"{'='*80}")
    
    from models.neural_renderer import NeuralRenderer
    
    prog_renderer = NeuralRenderer(
        bg_type=opts.bg_type,
        feat_nc=feat_nc,
        out_dim=3,
        final_actvn=True,
        min_feat=32,
        featmap_size=featmap_size,
        img_size=opts.pred_img_size,
    )
    
    with torch.no_grad():
        output_prog = prog_renderer(x, shape_code)
    
    print(f"Output shape: {output_prog.shape}")
    print(f"Output range: [{output_prog.min():.3f}, {output_prog.max():.3f}]")
    
    total_params_prog = sum(p.numel() for p in prog_renderer.parameters())
    print(f"Total parameters: {total_params_prog:,}")
    
    print(f"\n{'='*80}")
    print("Comparison Summary")
    print(f"{'='*80}")
    print(f"DiT parameters: {total_params:,}")
    print(f"Progressive parameters: {total_params_prog:,}")
    print(f"Parameter ratio (DiT/Progressive): {total_params/total_params_prog:.2f}x")
    
    assert output.shape == output_prog.shape, "Output shapes must match!"
    print(f"\n✓ Output shapes match: {output.shape}")
    
    print(f"\n{'='*80}")
    print("Testing GazeGaussianNet with DiT")
    print(f"{'='*80}")
    
    try:
        from models.gaze_gaussian import GazeGaussianNet
        
        print("\nInitializing GazeGaussianNet with DiT renderer...")
        net = GazeGaussianNet(opts)
        
        print(f"Neural renderer type: {type(net.neural_render).__name__}")
        
        if opts.neural_renderer_type == "dit":
            assert type(net.neural_render).__name__ == "DiTNeuralRenderer"
            print("✓ DiT renderer successfully integrated!")
        else:
            print("✓ Progressive renderer is default (change neural_renderer_type to 'dit' in config)")
            
    except Exception as e:
        print(f"✗ Error initializing GazeGaussianNet: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("All tests completed!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    test_dit_renderer()
