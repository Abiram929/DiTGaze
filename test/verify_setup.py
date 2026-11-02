import sys
import os
import torch

print("=" * 80)
print("GAZEGAUSSIAN SETUP VERIFICATION")
print("=" * 80)

errors = []
warnings = []
successes = []

def check(name, condition, error_msg="", warning=False):
    if condition:
        successes.append(f"✓ {name}")
        print(f"✓ {name}")
        return True
    else:
        if warning:
            warnings.append(f"⚠ {name}: {error_msg}")
            print(f"⚠ {name}: {error_msg}")
        else:
            errors.append(f"✗ {name}: {error_msg}")
            print(f"✗ {name}: {error_msg}")
        return False

print("\n" + "="*80)
print("1. CHECKING PYTHON ENVIRONMENT")
print("="*80)

check("Python version", sys.version_info >= (3, 7), f"Python 3.7+ required, found {sys.version}")

print("\n" + "="*80)
print("2. CHECKING PYTORCH & CUDA")
print("="*80)

try:
    import torch
    check("PyTorch installed", True)
    print(f"  PyTorch version: {torch.__version__}")
except ImportError as e:
    check("PyTorch installed", False, str(e))

check("CUDA available", torch.cuda.is_available(), "CUDA not available - training will fail")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU device: {torch.cuda.get_device_name(0)}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print("\n" + "="*80)
print("3. CHECKING REQUIRED PACKAGES")
print("="*80)

packages = [
    "numpy", "cv2", "h5py", "lpips", "torchvision",
    "PIL", "trimesh", "tqdm", "einops", "pytorch3d",
    "tensorboardX", "plyfile"
]

for pkg in packages:
    try:
        if pkg == "cv2":
            import cv2
        elif pkg == "PIL":
            from PIL import Image
        else:
            __import__(pkg)
        check(f"{pkg} installed", True)
    except ImportError:
        check(f"{pkg} installed", False, f"Run: pip install {pkg}")

print("\n" + "="*80)
print("4. CHECKING CUSTOM MODULES")
print("="*80)

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    check("diff-gaussian-rasterization", True)
except ImportError as e:
    check("diff-gaussian-rasterization", False, 
          "Run: cd submodules/diff-gaussian-rasterization && pip install .")

try:
    from simple_knn._C import distCUDA2
    check("simple-knn", True)
except ImportError as e:
    check("simple-knn", False, 
          "Run: cd submodules/simple-knn && pip install .")

print("\n" + "="*80)
print("5. CHECKING DIRECTORY STRUCTURE")
print("="*80)

required_dirs = [
    "configs",
    "configs/dataset",
    "configs/dataset/eth_xgaze",
    "configs/dataset/eth_xgaze/cam",
    "configs/config_files",
    "configs/config_models",
    "dataloader",
    "losses",
    "models",
    "trainer",
    "utils",
]

for dir_path in required_dirs:
    check(f"Directory: {dir_path}", os.path.exists(dir_path))

print("\n" + "="*80)
print("6. CHECKING REQUIRED FILES")
print("="*80)

required_files = [
    "train_gazegaussian.py",
    "configs/gazegaussian_options.py",
    "configs/dataset/eth_xgaze/face_model.txt",
    "configs/config_models/epoch_60_512_ckpt.pth.tar",
    "models/gaze_gaussian.py",
    "models/neural_renderer.py",
    "models/gaussian_model.py",
    "models/camera_module.py",
    "losses/gazenerf_loss.py",
    "trainer/gazegaussian_trainer.py",
]

for file_path in required_files:
    check(f"File: {file_path}", os.path.exists(file_path))

print("\n" + "="*80)
print("7. CHECKING DATA DIRECTORY")
print("="*80)

from configs.gazegaussian_options import BaseOptions
opt = BaseOptions()

check(f"Data directory: {opt.img_dir}", 
      os.path.exists(opt.img_dir),
      f"Create directory or update img_dir in configs/gazegaussian_options.py",
      warning=True)

if os.path.exists(opt.img_dir):
    h5_files = [f for f in os.listdir(opt.img_dir) if f.endswith('.h5')]
    check(f"H5 data files found: {len(h5_files)}", 
          len(h5_files) > 0,
          "No .h5 files found in data directory",
          warning=True)
    if len(h5_files) > 0:
        print(f"  Found {len(h5_files)} .h5 files")

print("\n" + "="*80)
print("8. CHECKING MODEL INITIALIZATION")
print("="*80)

try:
    from models.gaze_gaussian import GazeGaussianNet
    from configs.meshhead_options import BaseOptions as MeshHeadOptions
    
    check("Import GazeGaussianNet", True)
    
    if os.path.exists(opt.load_meshhead_checkpoint):
        print(f"  Found meshhead checkpoint: {opt.load_meshhead_checkpoint}")
    else:
        check("Meshhead checkpoint", False, 
              f"Checkpoint not found: {opt.load_meshhead_checkpoint}\n" +
              "  You need to either:\n" +
              "  1. Train meshhead first: python train_meshhead.py\n" +
              "  2. Update load_meshhead_checkpoint path in configs/gazegaussian_options.py",
              warning=False)
        
except Exception as e:
    check("Model initialization", False, str(e))

print("\n" + "="*80)
print("9. TESTING FORWARD PASS (DRY RUN)")
print("="*80)

if not errors:
    try:
        print("Creating dummy batch...")
        batch_size = 2
        
        dummy_data = {
            'image': torch.randn(batch_size, 3, 512, 512).cuda(),
            'face_mask': torch.ones(batch_size, 1, 512, 512).cuda(),
            'left_eye_mask': torch.zeros(batch_size, 1, 512, 512).cuda(),
            'right_eye_mask': torch.zeros(batch_size, 1, 512, 512).cuda(),
            'ldms': torch.randn(batch_size, 68, 2).cuda(),
            'ldms_3d': torch.randn(batch_size, 68 + 100, 3).cuda(),
            'cam_ind': torch.zeros(batch_size, 1).long().cuda(),
            'down_scale': torch.tensor([4.0]),
            'fovx': torch.tensor([1.0] * batch_size).cuda(),
            'fovy': torch.tensor([1.0] * batch_size).cuda(),
            'world_view_transform': torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda(),
            'projection_matrix': torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda(),
            'full_proj_transform': torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda(),
            'camera_center': torch.zeros(batch_size, 3).cuda(),
            'nl3dmm_para_dict': {
                'shape_code': torch.randn(batch_size, 144).cuda(),
                'iden_code': torch.randn(batch_size, 80).cuda(),
                'expr_code': torch.randn(batch_size, 64).cuda(),
                'pitchyaw': torch.randn(batch_size, 2).cuda(),
                'pose': torch.randn(batch_size, 6).cuda(),
                'scale': torch.ones(batch_size, 1).cuda(),
                'intrinsics': torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda(),
                'extrinsics': torch.eye(3, 4).unsqueeze(0).repeat(batch_size, 1, 1).cuda(),
            }
        }
        
        print("Testing gaussian_model.py batch processing...")
        from models.gaussian_model import GaussianModel
        
        xyz_test = torch.randn(1000, 3)
        feature_test = torch.randn(1000, 32)
        landmarks_test = torch.randn(68, 3)
        
        gaussian_test = GaussianModel(
            opt, xyz_test, feature_test, landmarks_test, 
            add_mouth_points=False, is_eye=True
        ).cuda()
        
        gaussians = gaussian_test.generate(dummy_data)
        check("Gaussian generation (batch=2)", gaussians['xyz'].shape[0] == batch_size)
        
        print("Testing neural_renderer.py...")
        from models.neural_renderer import NeuralRenderer
        
        renderer = NeuralRenderer(
            bg_type="white",
            feat_nc=32,
            out_dim=3,
            final_actvn=True,
            min_feat=32,
            featmap_size=128,
            img_size=512
        ).cuda()
        
        test_feat = torch.randn(batch_size, 32, 128, 128).cuda()
        test_shape = torch.randn(batch_size, 144).cuda()
        
        with torch.no_grad():
            output = renderer(test_feat, test_shape)
        
        check("Neural renderer forward pass", output.shape == (batch_size, 3, 512, 512))
        check("Neural renderer output range", 
              output.min() >= 0 and output.max() <= 1,
              f"Output range [{output.min():.3f}, {output.max():.3f}] not in [0, 1]")
        
        print("Testing loss computation...")
        from losses.gazenerf_loss import GazeNeRFLoss
        
        loss_utils = GazeNeRFLoss(
            gaze_loss_importance=0.1,
            eye_loss_importance=1.0,
            face_loss_importance=1.0,
            vgg_importance=0.1,
            use_vgg_loss=False,
            use_ssim_loss=True,
            use_l1_loss=True,
            use_angular_loss=False,
            device='cuda'
        )
        
        test_render_dict = {
            'merge_img': torch.rand(batch_size, 3, 512, 512).cuda(),
            'merge_img_pro': torch.rand(batch_size, 3, 512, 512).cuda(),
            'merge_img_face': torch.rand(batch_size, 3, 512, 512).cuda(),
            'merge_img_face_pro': torch.rand(batch_size, 3, 512, 512).cuda(),
            'merge_img_eyes': torch.rand(batch_size, 3, 512, 512).cuda(),
            'merge_img_eyes_pro': torch.rand(batch_size, 3, 512, 512).cuda(),
        }
        
        test_data_for_loss = {
            'total_render_dict': test_render_dict,
            'image': torch.rand(batch_size, 3, 512, 512).cuda(),
            'face_mask': torch.ones(batch_size, 1, 512, 512).cuda(),
            'left_eye_mask': torch.zeros(batch_size, 1, 512, 512).cuda(),
            'right_eye_mask': torch.zeros(batch_size, 1, 512, 512).cuda(),
            'cam_ind': torch.zeros(batch_size, 1).long().cuda(),
            'ldms': torch.randn(batch_size, 68, 2).cuda(),
            'opt_cam_dict': None,
            'opt_code_dict': {
                'iden': torch.zeros(batch_size, 80).cuda(),
                'expr': torch.zeros(batch_size, 64).cuda(),
            }
        }
        
        loss_dict = loss_utils.calc_total_loss(test_data_for_loss, epoch=0, batch_num=0)
        
        check("Loss computation", 'total_loss' in loss_dict)
        check("Loss is finite", torch.isfinite(loss_dict['total_loss']).item())
        
        print(f"  Test loss value: {loss_dict['total_loss'].item():.4f}")
        
        check("All forward pass tests", True)
        
    except Exception as e:
        check("Forward pass test", False, str(e))
        import traceback
        print("\nDetailed error:")
        traceback.print_exc()
else:
    print("⊗ Skipping forward pass test due to previous errors")

print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

print(f"\n✓ Successes: {len(successes)}")
print(f"⚠ Warnings: {len(warnings)}")
print(f"✗ Errors: {len(errors)}")

if warnings:
    print("\n⚠ WARNINGS:")
    for w in warnings:
        print(f"  {w}")

if errors:
    print("\n✗ ERRORS THAT MUST BE FIXED:")
    for e in errors:
        print(f"  {e}")
    print("\n⊗ CANNOT START TRAINING - Fix errors above first!")
    sys.exit(1)
else:
    print("\n" + "="*80)
    print("✓✓✓ ALL CHECKS PASSED - READY TO TRAIN ✓✓✓")
    print("="*80)
    print("\nYou can now start training with:")
    print("  python train_gazegaussian.py --batch_size 2 --num_epochs 10")
    print("\nOr test batch sizes:")
    print("  python test_batch_training.py")
    print("="*80)
