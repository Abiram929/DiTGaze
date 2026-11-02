import os
import sys
import torch

def print_status(message, status):
    symbols = {"pass": "✅", "fail": "❌", "warn": "⚠️", "info": "ℹ️"}
    print(f"{symbols.get(status, '•')} {message}")

def check_cuda():
    if torch.cuda.is_available():
        print_status(f"CUDA available: {torch.cuda.get_device_name(0)}", "pass")
        print_status(f"CUDA version: {torch.version.cuda}", "info")
        return True
    else:
        print_status("CUDA not available - training will be very slow!", "warn")
        return False

def check_file(filepath, description, required=True):
    exists = os.path.exists(filepath)
    if exists:
        print_status(f"{description}: {filepath}", "pass")
        return True
    else:
        status = "fail" if required else "warn"
        print_status(f"{description} missing: {filepath}", status)
        return False

def check_directory(dirpath, description, required=True):
    exists = os.path.isdir(dirpath)
    if exists:
        items = len(os.listdir(dirpath))
        print_status(f"{description}: {dirpath} ({items} items)", "pass")
        return True
    else:
        status = "fail" if required else "warn"
        print_status(f"{description} missing: {dirpath}", status)
        return False

def main():
    print("\n" + "="*60)
    print("GazeGaussian Training Readiness Check")
    print("="*60 + "\n")
    
    all_checks_pass = True
    
    print("🔍 Checking System Requirements...")
    print("-" * 60)
    if not check_cuda():
        all_checks_pass = False
    print()
    
    print("📦 Checking Critical Files...")
    print("-" * 60)
    
    critical_files = [
        ("configs/config_models/tets_data.npz", "Tetrahedral mesh data", True),
        ("configs/config_files/xgaze_evaluation_target_single_subject.txt", "Evaluation targets", True),
        ("configs/dataset/eth_xgaze/train_test_split.json", "Train/test split", True),
    ]
    
    for filepath, desc, required in critical_files:
        if not check_file(filepath, desc, required):
            if required:
                all_checks_pass = False
    print()
    
    print("📂 Checking Data Directories...")
    print("-" * 60)
    
    data_dirs = [
        ("data/ETH-XGaze", "Training data", True),
        ("configs/config_models", "Model configuration files", True),
        ("configs/config_files", "Config files", True),
    ]
    
    for dirpath, desc, required in data_dirs:
        if not check_directory(dirpath, desc, required):
            if required:
                all_checks_pass = False
    print()
    
    print("🎯 Checking Training Configurations...")
    print("-" * 60)
    
    try:
        from configs.gazegaussian_options import BaseOptions as GazeOptions
        opts = GazeOptions()
        print_status("GazeGaussian options loaded", "pass")
        
        if opts.load_meshhead_checkpoint:
            if os.path.exists(opts.load_meshhead_checkpoint):
                print_status(f"MeshHead checkpoint: {opts.load_meshhead_checkpoint}", "pass")
            else:
                print_status(f"MeshHead checkpoint not found: {opts.load_meshhead_checkpoint}", "warn")
                print("   → You'll need to train meshhead first or download a pretrained checkpoint")
        else:
            print_status("No MeshHead checkpoint set - will need one to train", "warn")
            
        if opts.load_gazegaussian_checkpoint:
            if os.path.exists(opts.load_gazegaussian_checkpoint):
                print_status(f"GazeGaussian checkpoint: {opts.load_gazegaussian_checkpoint}", "pass")
            else:
                print_status(f"GazeGaussian checkpoint not found: {opts.load_gazegaussian_checkpoint}", "fail")
                all_checks_pass = False
        else:
            print_status("No GazeGaussian checkpoint set - will train from scratch", "info")
            
        print_status(f"Batch size: {opts.batch_size}", "info")
        print_status(f"Num epochs: {opts.num_epochs}", "info")
        print_status(f"Learning rate: {opts.lr}", "info")
        
    except Exception as e:
        print_status(f"Failed to load config: {str(e)}", "fail")
        all_checks_pass = False
    print()
    
    print("🔧 Checking Submodules...")
    print("-" * 60)
    
    try:
        import diff_gaussian_rasterization
        print_status("diff-gaussian-rasterization installed", "pass")
    except ImportError:
        print_status("diff-gaussian-rasterization not installed", "fail")
        print("   → Run: cd submodules/diff-gaussian-rasterization && pip install -e .")
        all_checks_pass = False
        
    try:
        import simple_knn
        print_status("simple-knn installed", "pass")
    except ImportError:
        print_status("simple-knn not installed", "fail")
        print("   → Run: cd submodules/simple-knn && pip install -e .")
        all_checks_pass = False
    print()
    
    print("="*60)
    if all_checks_pass:
        print("✅ All critical checks passed! Ready to train.")
        print("\nNext steps:")
        print("  1. For MeshHead: python train_meshhead.py --batch_size 1 --num_epochs 10")
        print("  2. For GazeGaussian: python train_gazegaussian.py --batch_size 1 --num_epochs 100")
        print("     (Make sure to set load_meshhead_checkpoint in config first)")
    else:
        print("❌ Some critical checks failed. Please fix the issues above before training.")
        print("\nQuick fixes:")
        print("  • Missing config_models: Download from HuggingFace (see README)")
        print("  • Missing data: Check data directory path in config")
        print("  • Missing submodules: Install from submodules/ directory")
        return 1
    
    print("="*60 + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
